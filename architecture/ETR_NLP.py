import torch
import torch.nn as nn
from architecture.abstract_arch import AbsArchitecture
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import  global_max_pool


class Encoder(nn.Module):
    def __init__(self, task_name, in_features, dropout, emb_size):
        super(Encoder, self).__init__()
        prims = ['GCN', 'GIN', 'GAT', 'dilated_gconv', 'perturbation']
        gamma = 0.9
        self.etr_nlp1 = ETR_NLP_Block(
            in_features, out_features=64, 
            prims=prims, gamma=gamma, task_names=task_name
            )
        self.etr_nlp2 = ETR_NLP_Block(
            in_features=64, out_features=128, 
            prims=prims, gamma=gamma, task_names=task_name
            )
        self.etr_nlp3 = ETR_NLP_Block(
            in_features=128, out_features=emb_size, 
            prims=prims, gamma=gamma, task_names=task_name
            )
        self.hidden = nn.Linear(emb_size, emb_size)
        self.bn = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, task_name):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.etr_nlp1(x, task_name, edge_index)
        x = self.etr_nlp2(x, task_name, edge_index)
        x = self.etr_nlp3(x, task_name, edge_index)
        x = global_max_pool(x, batch)
        x = self.hidden(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    

class ETR_NLP_Block(nn.Module):
    def __init__(self, in_features, out_features, prims, gamma, task_names, **kwargs):
        super(ETR_NLP_Block, self).__init__()
        features_shared = int(out_features * gamma)
        features_specif = int(out_features - features_shared)
        self.task_names = task_names
        # define the shared branch
        self.shared_brach = NLP(
            in_features=in_features, out_features=features_shared, 
            prims=prims, task_names=task_names, **kwargs
            )
        self.shared_bn = nn.BatchNorm1d(features_shared)
        # define the specif brach
        self.sepcif_brach = nn.ModuleDict(
            {task: GCNConv(in_features, features_specif) for task in task_names}
            )
        self.specif_bn = nn.BatchNorm1d(features_specif)
        self.relu = nn.ReLU()

    def forward(self, data, task_name, edge_index):
        shared = self.shared_brach(data, edge_index, task_name)
        shared = self.shared_bn(shared)
        shared = self.relu(shared)
        specif = self.sepcif_brach[task_name](data, edge_index)
        specif = self.specif_bn(specif)
        specif = self.relu(specif)
        total_features = torch.cat([shared, specif], dim = 1)
        return total_features


class NLP(nn.Module):
    def __init__(self, in_features, out_features, prims, task_names, task_specif_fc=False, **kwargs):
        super(NLP, self).__init__()
        # define non-learnable primitives
        self.prims = prims
        # the number of NLPs
        k = len(prims)
        # GCN
        if 'GCN' in self.prims:
            self.gcn = GCNConv(in_features, in_features)
            self.gcn.requires_grad = False
        # GIN
        if 'GIN' in self.prims:
            # self.gin_mlp = nn.Sequential(nn.Linear(in_features, in_features))
            self.gin_mlp = nn.Linear(in_features, in_features)
            self.gin_mlp.requires_grad = False
            self.gin = GINConv(nn=self.gin_mlp)
            self.gin.requires_grad = False
        #GAT
        if 'GAT' in self.prims:
            self.gat = GATConv(in_features, in_features)
            self.gat.requires_grad = False
        # dilated gconv
        if 'dilated_gconv' in self.prims:
            self.dilatedconv = GCNConv(in_features, in_features, dilation=2)
            self.dilatedconv.requires_grad = False
        # perturbation
        if 'perturbation' in self.prims:
            self.perturb = AddUniformPerturbation(scale=0.1)
            self.perturb.requires_grad = False
        # whether linear compressed layer in each task or just one?
        self.task_specif_fc = task_specif_fc
        if self.task_specif_fc:
            self.compressed = nn.ModuleDict({task: nn.Linear(in_features * k, out_features) for task in task_names})
        else:
            self.linear = nn.Linear(in_features*k, out_features)

    def forward(self, data, edge_index, task_name):
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        # define the list to concat
        cat_list = []
        # extract features by NLPs
        if 'GCN' in self.prims:
            GCN_feature = self.gcn(data, edge_index)
            cat_list.append(GCN_feature)
        if 'GIN' in self.prims:
            GIN_feature = self.gin(data, edge_index)
            cat_list.append(GIN_feature)
        if 'GAT' in self.prims:
            GAT_feature = self.gat(data, edge_index)
            cat_list.append(GAT_feature)
        if 'dilated_gconv' in self.prims:
            dilated_feature = self.dilatedconv(data, edge_index)
            cat_list.append(dilated_feature)
        if 'perturbation' in self.prims:
            perturb_feature = self.perturb(data)
            cat_list.append(perturb_feature)

        x = torch.cat(cat_list, dim=1)
        # linear player to compress the features
        if self.task_specif_fc:
            x = self.compressed[task_name](x)
        else:
            x = self.linear(x)
        return x
    
# define uniform distribution perturbative function
class AddUniformPerturbation(nn.Module):
    def __init__(self, scale):
        super(AddUniformPerturbation, self).__init__()
        self.scale = scale
        #  use scale to control the magnitude of perturbation
    def forward(self, data):
        data = data + torch.randn_like(data) * self.scale
        return data
    
class ETR_NLP(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, **kwargs):
        super(ETR_NLP, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.encoder = encoder_class(task_name, in_features=96, dropout=0.1, emb_size=256)
        self.decoders = decoders

    def forward(self, data, task_name):
        x = self.encoder(data, task_name)
        emb = {task_name: x}
        out = {task_name: self.decoders[task_name](x)}
        return out, emb