import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_max_pool

from architecture.abstract_arch import AbsArchitecture


class Encoder(nn.Module):
    def __init__(self, in_features = 96, dropout = 0.1, emb_size=256):
        super(Encoder, self).__init__()
        self.gconv1 = GCNConv(in_channels = in_features, out_channels = 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.gconv2 = GCNConv(in_channels = 64, out_channels = 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.gconv3 = GCNConv(in_channels = 128, out_channels = 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.hidden = nn.Linear(in_features = 256, out_features = emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, layer_idx):
        if layer_idx == 0:
            x, self.edge_index, self.batch = data.x, data.edge_index, data.batch
            x = self.gconv1(x, self.edge_index)
            x = self.bn1(x)
            x = self.relu(x)
            return x
        elif layer_idx == 1:
            x = self.gconv2(data, self.edge_index)
            x = self.bn2(x)
            x = self.relu(x)
            return x
        elif layer_idx == 2:
            x = self.gconv3(data, self.edge_index)
            x = self.bn3(x)
            x = self.relu(x)
            return x
        elif layer_idx == 3:
            x = global_max_pool(data, self.batch)
            x = self.hidden(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x

class transform_ltb(nn.Module):
    def __init__(self, encoder_list, task_name, max_epochs, device):
        super(transform_ltb, self).__init__()
        self.device = device
        self.task_name = task_name
        self.task_num = len(task_name)
        # layers 0
        self.rep_encoder = nn.ModuleDict({task: encoder_list() for task in task_name})
        self.alpha = nn.Parameter(torch.ones(2, self.task_num, self.task_num))
        self.max_epochs = max_epochs
        self.current_epoch = 0

    def forward(self, inputs):
        if self.current_epoch == 0: # warm-up
            alpha = torch.ones(2, self.task_num, self.task_num).to(self.device)
            self.current_epoch += 1
        else:
            tau = self.max_epochs/20 / np.sqrt(self.current_epoch+1)
            alpha = F.gumbel_softmax(self.alpha, dim=-1, tau=tau, hard=True)

        ss_rep = {i:[0]*self.task_num for i in range(4)}
        # Task-cross 一共有3层，每层每个任务有自己的encoder，α控制第i-1层的数据进入第i层时，每个任务的encoder加和的权重，实现跨任务交互
        for i in range(4):
            for tn, task in enumerate(self.task_name):
                if i == 0:
                    ss_rep[i][tn] = self.rep_encoder[task](inputs, layer_idx=0)
                elif i > 0 and i < 3:
                    child_rep = sum([alpha[i-1, tn, j] * ss_rep[i-1][j] for j in range(self.task_num)])
                    ss_rep[i][tn] = self.rep_encoder[task](child_rep, layer_idx=i)
                else:
                    child_rep = ss_rep[i-1][tn]
                    ss_rep[i][tn] = self.rep_encoder[task](child_rep, layer_idx=i)
        return ss_rep

class LTB(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, **kwargs):
        super(LTB, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.task_name = task_name
        self.task_encoder = transform_ltb(encoder_list=encoder_class, task_name=task_name, max_epochs=100, device=device)

    def forward(self, inputs, task_name = None):
        ss_rep = self.task_encoder(inputs)
        task_name_idx = dict(zip(self.task_name, range(len(self.task_name))))
        current_task_idx = task_name_idx[task_name]
        emb = {task:ss_rep[3][current_task_idx] for task in self.task_name}
        pred = self.decoders[task_name](ss_rep[3][current_task_idx])
        out = {task: pred for task in self.task_name}
        return out, emb