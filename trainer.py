import copy
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from record import PerformanceMeter
from utils import count_parameters

from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from architecture.Graphormer import Graphormer
from sampling.LLoss import LossNet
from sampling.TiDAL import TDNet

import algos


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DataCollator(nn.Module):
    def __init__(self, max_node, spatial_pos_max, device):
        super(DataCollator, self).__init__()
        self.max_node = max_node # 最大节点数
        self.spatiak_pos_max = spatial_pos_max # 最大空间位置
        self.device = device

    def split_Batch(self, batch):
        # Split a batch graph to single graph.
        num_graphs = batch.batch.max().item() + 1
        data_list = []
        for i in range(num_graphs):
            mask = batch.batch == i
            edge_mask = mask[batch.edge_index[0]]
            # Node and edge of single graph
            sub_x = batch.x[mask]
            sub_edge_index = batch.edge_index[:, edge_mask]
            sub_y = batch.y[i] if batch.y is not None else None
            # Re-project
            node_idx = torch.zeros_like(mask, dtype = torch.long)
            node_idx[mask] = torch.arange(mask.sum())
            sub_edge_index = node_idx[sub_edge_index]
            # Create new Data
            data = Data(
                x = sub_x, 
                edge_index = sub_edge_index, 
                y = sub_y
                )
            data_list.append(data)
        return data_list

    def preprocess_item(self, item):
        edge_index, x = item.edge_index, item.x
        x = self.convert_to_single_emb(x)
        N = x.size(0)
        # Centrality Encoding (Adj matrix -> Degree matrix)
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True
        # Spatial Encoding (Calculate shortest path via Floyd algorithm)
        result, path = algos.floyd_warshall(adj.numpy())
        spatial_pos = torch.from_numpy((result)).long()
        # Attention bias (adding VNode)
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
        # Combine
        item.x = x
        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree  # for undirected graph
        item.spatial_pos = spatial_pos
        item.attn_bias = attn_bias
        return item

    @staticmethod
    def convert_to_single_emb(x, offset = 1):
        feature_num = x.size(1) if len(x.size()) > 1 else 1
        feature_offset = 1 + torch.arange(0, feature_num * offset, offset)
        x = x + feature_offset
        return x
    
    def pad_1d_unsqueeze(self, x, padlen):
        x = x + 1  # pad id = 0
        xlen = x.size(0)
        if xlen < padlen:
            new_x = x.new_zeros([padlen], dtype = x.dtype)
            new_x[:xlen] = x
            x = new_x
        return x.unsqueeze(0)

    def pad_2d_unsqueeze(self, x, padlen):
        x = x + 1  # pad id = 0
        xlen, xdim = x.size()
        if xlen < padlen:
            new_x = x.new_zeros([padlen, xdim], dtype = x.dtype)
            new_x[:xlen, :] = x
            x = new_x
        return x.unsqueeze(0)

    def pad_attn_bias_unsqueeze(self, x, padlen):
        xlen = x.size(0)
        if xlen < padlen:
            new_x = x.new_zeros([padlen, padlen], dtype = x.dtype).fill_(float("-inf"))
            new_x[:xlen, :xlen] = x
            new_x[xlen:, :xlen] = 0
            x = new_x
        return x.unsqueeze(0)

    def pad_spatial_pos_unsqueeze(self, x, padlen):
        x = x + 1
        xlen = x.size(0)
        if xlen < padlen:
            new_x = x.new_zeros([padlen, padlen], dtype = x.dtype)
            new_x[:xlen, :xlen] = x
            x = new_x
        return x.unsqueeze(0)

    def collator(self, items):
        max_node = self.max_node
        spatial_pos_max = self.spatiak_pos_max

        items = [item for item in items if item is not None and item.x.size(0) <= self.max_node]
        items = [(
            item.x,
            item.in_degree,
            item.out_degree,
            item.spatial_pos,
            item.attn_bias,
            item.y) for item in items]
        (xs, in_degrees, out_degrees, spatial_poses, attn_biases, ys) = zip(*items)
        max_node_num = max(i.size(0) for i in xs)

        x = torch.cat([self.pad_2d_unsqueeze(i, max_node_num) for i in xs])
        degree = torch.cat([self.pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
        spatial_pos = torch.cat([self.pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses])
        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        attn_bias = torch.cat([self.pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
        y = torch.cat(ys)

        data_to_model = Data(
            x = x.long(),
            in_degree = degree,
            out_degree = degree,
            spatial_pos = spatial_pos,
            attn_bias = attn_bias,
            y = y.view(-1,1),
            )
        return data_to_model

    def move_dict_to_gpu(self, data_dict, device):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)
            elif isinstance(value, dict):
                data_dict[key] = self.move_dict_to_gpu(value, device)
            elif isinstance(value, list):
                data_dict[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        return data_dict

    def forward(self, batch):
        # Split
        data_list = self.split_Batch(batch)
        # Centrality Encoding & Spatial Encoding
        items = [self.preprocess_item(data) for data in data_list]
        # Collate and move to GPU
        items = self.collator(items)
        data_to_Model = self.move_dict_to_gpu(items, device=self.device)
        return data_to_Model
    

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.
    '''
    def __init__(self, task_dict, weighting, architecture, sampling, 
                 encoder_class, decoders, optim_param, 
                 save_path=None, load_path=None, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device(int(kwargs.get('gpu_id', 'cpu')))
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.sampling = sampling.__name__   # 保存sampling方法名称
        self.save_path = save_path
        self.load_path = load_path

        self.prepare_model(weighting, architecture, sampling, encoder_class, decoders)
        self.prepare_optimizer(optim_param)

        self.meter = PerformanceMeter(self.task_dict)

        # DataCollator for Transformer-based architecture
        if architecture in [Graphormer]:
            self.dataCollator = DataCollator(max_node=512, spatial_pos_max=20, device=self.device)
        else:
            self.dataCollator = torch.nn.Identity()

    def prepare_model(self, weighting, architecture, sampling, encoder_class, decoders):
        class MTLmodel(architecture, weighting, sampling):
            def __init__(self, task_name, encoder_class, decoders, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        
        # 根据方法选择不同的副网络，并移动到GPU
        if self.sampling == 'LLoss':
            self.uncertainty_estimator = LossNet(task_name=self.task_name).to(self.device)
        elif self.sampling == 'TiDAL':
            self.uncertainty_estimator = TDNet(task_name=self.task_name).to(self.device)

        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))

        count_parameters(self.model)
        
    def prepare_optimizer(self, optim_param):
        optim_dict = {'adam': torch.optim.Adam}
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        # 副网络优化器 
        if self.sampling in ['LLoss', 'TiDAL']:
            self.uncertainty_optimizer = optim_dict[optim_param['optim']](self.uncertainty_estimator.parameters(), **optim_arg)

    def prepare_dataloaders(self, dataloaders):
        loader = {}
        batch_num = []
        for task in self.task_name:
            loader[task] = dataloaders[task]
            batch_num.append(len(dataloaders[task]))
        return loader, batch_num
    
    def process_data(self, loader):
        try:
            loader = iter(loader)
            data = next(loader)
        except:
            data = next(loader)
        return data

    def compute_loss(self, preds, gts, task_name=None):
        train_losses = self.meter.losses[task_name].update_loss(preds, gts)
        return train_losses

    def compute_mdn_loss(self, preds, gts, task_name=None):
        train_losses = self.meter.losses[task_name].update_mdn_loss(preds, gts)
        return train_losses
    
    def compute_sample_loss(self, preds, gts, task_name=None):
        sample_losses = self.meter.losses[task_name].compute_sample_loss(preds, gts)
        return sample_losses
    
    def train(self, task_type, switch, sampling, 
              dataset, n_total, label_idx, unlabel_idx, val_idx, test_idx, 
              batch_size, epochs, n_query):
        r'''The training process of multi-task learning.
        '''
        # 确定batchsize
        for task, idx in label_idx.items():
            if batch_size > len(idx):
                batch_size = len(idx)
        # 包装trainloader
        train_loader = {}
        for task, idx in label_idx.items():
            train_loader[task] = DataLoader(
                dataset = dataset[task], 
                batch_size = batch_size,
                sampler=SubsetRandomSampler(label_idx[task])
                )

        train_loader, train_batch = self.prepare_dataloaders(train_loader)
        train_batch = max(train_batch)
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs

        self.meter.init_display()
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()

            if self.sampling in ['LLoss', 'TiDAL']:
                self.uncertainty_estimator.train()  # 确保副模型也处于训练模式

            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                train_losses = torch.zeros(self.task_num).to(self.device)
                for tn, task in enumerate(self.task_name):
                    # train_input[0]:data, train_input[1]:idxs
                    train_data, batch_idxs = self.process_data(train_loader[task])
                    train_data = self.dataCollator(train_data)
                    train_data = train_data.to(self.device)
                    train_pred, train_embd = self.model(train_data, task)
                    train_pred = train_pred[task]
                    if self.sampling == 'PMAL':
                        train_losses[tn] = self.compute_mdn_loss(train_pred, train_data.y, task)
                    else:
                        train_losses[tn] = self.compute_loss(train_pred, train_data.y, task)

                    if sampling == 'LLoss':
                        if epoch > 0.2 * epochs:
                            features = train_embd[task].detach()
                        else:
                            features = train_embd[task]
                        # 计算当前任务预测损失
                        pred_loss = self.uncertainty_estimator(features, task)
                        target = self.compute_sample_loss(train_pred, train_data.y, task)
                        module_loss = self.uncertainty_estimator.loss_forward(pred_loss, target)
                        # 将副模型的损失加到主模型损失上
                        train_losses[tn] = train_losses[tn] + 1.0 * module_loss # 参数可调，这里写死了
                    if sampling == 'TiDAL':
                        if epoch > 0.2 * epochs:
                            # 使用倒数第二层特征作为表示
                            features = train_embd[task].detach()
                        else:
                            features = train_embd[task]
                        # 更新移动平均特征，挪到gpu上
                        moving_probs = dataset[task].moving_probs[batch_idxs,:].to(self.device)
                        moving_probs = (moving_probs * epoch + features) / (epoch + 1)  # 遵循原文的滑动平均计算公式
                        # 更新数据集中的 moving_probs
                        dataset[task].moving_probs[batch_idxs,:] = moving_probs.detach().cpu()
                        # 使用副模型预测特征分布
                        pred_feat = self.uncertainty_estimator(features, task)
                        # 使用KL散度计算损失 
                        pred_log_prob = F.log_softmax(pred_feat, dim=1)  # 预测特征的对数概率
                        moving_prob = F.softmax(moving_probs, dim=1).detach()  # 移动平均特征的概率
                        module_loss = nn.KLDivLoss(reduction='batchmean')(pred_log_prob, moving_prob)
                        train_losses[tn] = train_losses[tn] + 1.0 * module_loss # 参数可调，这里写死了

                    self.meter.update(train_pred, train_data.y, task)
                
                self.optimizer.zero_grad()
                if sampling in ['LLoss', 'TiDAL']:
                    self.uncertainty_optimizer.zero_grad()

                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w

                self.optimizer.step()
                if sampling in ['LLoss', 'TiDAL']:
                    self.uncertainty_optimizer.step()

            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_idx is not None:
                self.test(dataset, val_idx, epoch, mode='val')
            if test_idx is not None:
                self.test(dataset, test_idx, epoch, mode='test')

            if self.save_path is not None and self.meter.best_val_epoch['average'][0] == epoch:
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        
        self.meter.display_best_result()

        if switch == 'on':
            label_embds = self.predict_embedding(dataset, n_total, label_idx) if sampling in [
                'Core_set', 'CDAL', 'ProbCover'] else None
            unlabel_embds = self.predict_embedding(dataset, n_total, unlabel_idx) if sampling in [
                'Entropy', 'K_Means', 'Core_set', 'CDAL', 'BADGE', 'ProbCover'] else None
            unlabel_drops = self.predict_dropout(dataset, n_total, unlabel_idx) if sampling in [
                'BALD'] else None
            if sampling in ['LLoss', 'TiDAL']:
                unlabel_uncertainties = self.predict_uncertainty(dataset, n_total, unlabel_idx)
            elif sampling in ['PMAL']:
                unlabel_uncertainties = self.predict_mdn_uncertainty(dataset, n_total, unlabel_idx, task_type) 
            else:
                unlabel_uncertainties = None

            label_idx, unlabel_idx = self.model.query(
                n_total, n_query, label_idx, unlabel_idx, 
                label_embds, unlabel_embds, 
                unlabel_drops, unlabel_uncertainties)

        return label_idx, unlabel_idx

    def test(self, dataset, test_idx, epoch=None, mode='test'):
        r'''The test process of multi-task learning.
        '''
        # 包装testloader
        test_loader = {}
        for task, idx in test_idx.items():
            test_loader[task] = DataLoader(
                dataset = dataset[task], 
                batch_size = 1,
                sampler = SubsetSequentialSampler(test_idx[task])
                )
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                for batch_index, test_input in enumerate(test_loader[task]):
                    # test_input[0]:data, test_input[1]:idxs
                    test_input = self.dataCollator(test_input[0])
                    test_input = test_input.to(self.device)
                    test_pred, test_embd = self.model(test_input, task)
                    test_pred = test_pred[task]
                    self.meter.step(test_pred, test_input.y, task)
                    if batch_index == len(test_loader[task]) - 1:
                        preds = torch.cat(self.meter.caches[task]['preds'], dim = 0)
                        gts = torch.cat(self.meter.caches[task]['gts'], dim = 0)
                        self.compute_loss(preds, gts, task)
                        self.meter.update(preds, gts, task)
        
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        self.meter.reinit()

    def visualization(self, dataset, test_idx):
        r'''The visualization of multi-task model.
        '''
        # 包装testloader
        test_loader = {}
        for task, idx in test_idx.items():
            test_loader[task] = DataLoader(
                dataset = dataset[task], 
                batch_size = 1,
                sampler = SubsetSequentialSampler(test_idx[task])
                )
        
        self.model.eval()
        all_embds, all_preds, all_labels, all_uncertainties = {}, {}, {}, {}
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                all_embds[task] = torch.zeros([len(test_loader[task]), 256])
                all_preds[task] = torch.zeros([len(test_loader[task]), 1])
                all_labels[task] = torch.zeros([len(test_loader[task]), 1])
                all_uncertainties[task] = torch.zeros([len(test_loader[task]), 1])
                for batch_index, test_input in enumerate(test_loader[task]):
                    # test_input[0]:data, test_input[1]:idxs
                    test_input = self.dataCollator(test_input[0])
                    test_input = test_input.to(self.device)
                    test_pred, test_embd = self.model(test_input, task)
                    if len(test_pred[task]) == 3:
                        pi, mu, var = test_pred[task]
                        all_preds[task][batch_index] = torch.sum(pi.unsqueeze(-1) * mu, dim=1).view(-1, 1).cpu()
                        
                        total_mu = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
                        aleatoric_uncertainty = torch.sum(pi.unsqueeze(-1) * var, dim=1).squeeze(-1)
                        diff = mu - total_mu.unsqueeze(1)
                        diff_squared = torch.sum(diff**2, dim=-1)
                        epistemic_uncertainty = torch.sum(pi * diff_squared, dim=1)

                        all_uncertainties[task][batch_index] = (0.5*epistemic_uncertainty + 0.5*aleatoric_uncertainty).cpu()
                    else:
                        all_preds[task][batch_index] = test_pred[task].cpu()
                    all_embds[task][batch_index] = test_embd[task].cpu()
                    all_labels[task][batch_index] = test_input.y.cpu()

        from visual import merge_by_species
        all_preds = merge_by_species(all_preds)
        all_uncertainties = merge_by_species(all_uncertainties)
        all_embds = merge_by_species(all_embds)
        all_labels = merge_by_species(all_labels)

        from visual import plot_predictions_vs_actual_scatter
        plot_predictions_vs_actual_scatter(all_preds, all_labels)

        from visual import plot_predictions_vs_actual_dist
        plot_predictions_vs_actual_dist(all_preds, all_labels)

        from visual import plot_tsne_scatter
        plot_tsne_scatter(all_embds, all_labels)

        from visual import plot_uncertainty_kde
        plot_uncertainty_kde(all_uncertainties)



    def predict_embedding(self, dataset, n_total, test_idx):
        r'''The predict_embedding process of active learning.
        '''
        # 包装testloader, 注意, 这里使用SubsetSequentialSampler以保证不shuffle!
        test_loader = {}
        for task, idx in test_idx.items():
            test_loader[task] = DataLoader(
                dataset = dataset[task], 
                batch_size = 1,
                sampler = SubsetSequentialSampler(test_idx[task])
                )

        embds = {}
        self.model.eval()
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                embds[task] = torch.zeros([n_total[task], 256])
                for batch_index, test_input in enumerate(test_loader[task]):
                    # test_input[0]:data, test_input[1]:idxs
                    idxs = test_input[1].to(self.device)
                    test_input = self.dataCollator(test_input[0])
                    input = test_input.to(self.device)
                    test_pred, test_embd = self.model(input, task)
                    embds[task][idxs] = test_embd[task].cpu()
        return embds
    
    def predict_dropout(self, dataset, n_total, test_idx):
        r'''The predict_dropout process of active learning.
        '''
        # 包装testloader, 注意, 这里使用SubsetSequentialSampler以保证不shuffle!
        test_loader = {}
        for task, idx in test_idx.items():
            test_loader[task] = DataLoader(
                dataset = dataset[task], 
                batch_size = 1,
                sampler = SubsetSequentialSampler(test_idx[task])
                )

        n_drop = 5
        drops = {}
        self.model.train()
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                drops[task] = torch.zeros([n_drop, n_total[task], 256])
                for n in range(n_drop):
                    for batch_index, test_input in enumerate(test_loader[task]):
                        idxs = test_input[1].to(self.device)
                        test_input = self.dataCollator(test_input[0])
                        input = test_input.to(self.device)
                        test_pred, test_embd = self.model(input, task)
                        drops[task][n][idxs] = test_embd[task].cpu()
        return drops
    
    def predict_uncertainty(self, dataset, n_total, test_idx):
        """计算每个未标记样本的不确定性，支持不同方法"""
        uncertainties = {}
        self.model.eval()
        self.uncertainty_estimator.eval()
        test_loader = {}
        with torch.no_grad():
            for task, idx in test_idx.items():
                test_loader[task] = DataLoader(
                    dataset=dataset[task],
                    batch_size=1,
                    sampler=SubsetSequentialSampler(idx)
                )
                
                if self.sampling == 'LLoss':
                    # LearningLoss方法：直接预测损失值
                    uncertainty = torch.zeros(n_total[task])
                    for data in test_loader[task]:
                        inputs = data[0].to(self.device)
                        idx = data[1].to(self.device)
                        _, features = self.model(inputs, task)

                        pred_loss = self.uncertainty_estimator(features[task], task)
                        uncertainty[idx] = pred_loss.cpu()
                    uncertainties[task] = uncertainty
                if self.sampling == 'TiDAL':
                    # TiDAL方法：
                    uncertainty = torch.zeros(n_total[task])
                    for data in test_loader[task]:
                        inputs = data[0].to(self.device)
                        idx = data[1].to(self.device)
                        _, features = self.model(inputs, task)
                        
                        pred_feat = self.uncertainty_estimator(features[task], task)
                        prob = F.softmax(pred_feat, dim=-1)
                        uncertainty[idx] = -(prob * torch.log(prob + 1e-8)).sum(dim=1).cpu()
                    uncertainties[task] = uncertainty
                    
        return uncertainties
    
    def predict_mdn_uncertainty(self, dataset, n_total, test_idx, task_type):
        if task_type == 'cls':
            # 包装testloader, 注意, 这里使用SubsetSequentialSampler以保证不shuffle!
            test_loader = {}
            for task, idx in test_idx.items():
                test_loader[task] = DataLoader(
                    dataset = dataset[task], 
                    batch_size = 1,
                    sampler = SubsetSequentialSampler(test_idx[task])
                    )

            probs = {}
            uncertainties = {}
            self.model.eval()
            with torch.no_grad():
                for tn, task in enumerate(self.task_name):
                    probs[task] = torch.zeros([n_total[task], 5])
                    uncertainties[task] = torch.zeros([n_total[task], 1])
                    for batch_index, test_input in enumerate(test_loader[task]):
                        # test_input[0]:data, test_input[1]:idxs
                        idxs = test_input[1].to(self.device)
                        test_input = self.dataCollator(test_input[0])
                        input = test_input.to(self.device)
                        test_pred, test_embd = self.model(input, task)

                        pi, mu, var = test_pred[task]
                        for t in range(5):
                            # 采样分量索引k*
                            k_star = torch.multinomial(pi, num_samples=1).squeeze(1)
                            eps = torch.randn_like(mu)
                            # 采样被选中的分量
                            pred_k_star = mu[:,k_star,:] + torch.sqrt(var[:,k_star,:]) * eps[:,k_star,:]
                            probs[task][idxs, t] = F.sigmoid(pred_k_star.view(-1)).cpu()

                    # 1. 计算总不确定性 (预测熵) H[y|x,D_train]
                    mean_probs = torch.mean(probs[task], dim=-1)
                    total_uncertainty = - (mean_probs * torch.log(mean_probs + 1e-8) + (1 - mean_probs) * torch.log(1 - mean_probs + 1e-8))
                    # 2. 计算偶然不确定性 (期望熵) E_{p(ω|D_train)}[H[y|x,ω]]
                    sample_entropy = - (probs[task] * torch.log(probs[task] + 1e-8) + (1 - probs[task]) * torch.log(1 - probs[task] + 1e-8))
                    aleatoric_uncertainty = torch.mean(sample_entropy, dim=-1)
                    # 3. 计算认知不确定性 (互信息) I[y,ω|x,D_train]
                    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

                    uncertainties[task] = (0.5*epistemic_uncertainty+0.5*aleatoric_uncertainty).cpu()


        if task_type == 'reg':
            # 包装testloader, 注意, 这里使用SubsetSequentialSampler以保证不shuffle!
            test_loader = {}
            for task, idx in test_idx.items():
                test_loader[task] = DataLoader(
                    dataset = dataset[task], 
                    batch_size = 1,
                    sampler = SubsetSequentialSampler(test_idx[task])
                    )

            uncertainties = {}
            self.model.eval()
            with torch.no_grad():
                for tn, task in enumerate(self.task_name):
                    uncertainties[task] = torch.zeros([n_total[task]])
                    for batch_index, test_input in enumerate(test_loader[task]):
                        # test_input[0]:data, test_input[1]:idxs
                        idxs = test_input[1].to(self.device)
                        test_input = self.dataCollator(test_input[0])
                        input = test_input.to(self.device)
                        test_pred, test_embd = self.model(input, task)

                        pi, mu, var = test_pred[task]
                        # 1. 计算混合分布的均值
                        total_mu = torch.sum(pi.unsqueeze(-1) * mu, dim=1)  # [batch_size, 1]
                        # 2. 计算偶然不确定性 u_al = Σ_{k=1}^{K} π^k Σ^k
                        aleatoric_uncertainty = torch.sum(pi.unsqueeze(-1) * var, dim=1).squeeze(-1)  # [batch_size]
                        # 3. 计算认知不确定性 u_ep = Σ_{k=1}^{K} π^k ||μ^k - Σ_{i=1}^{K} π^i μ^i||^2
                        diff = mu - total_mu.unsqueeze(1)  # [batch_size, K, 1]
                        diff_squared = torch.sum(diff**2, dim=-1)  # [batch_size, K]
                        epistemic_uncertainty = torch.sum(pi * diff_squared, dim=1)  # [batch_size]
                        
                        uncertainties[task][idxs] = (0.5*epistemic_uncertainty+0.5*aleatoric_uncertainty).cpu()

        return uncertainties