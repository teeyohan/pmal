import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from architecture.abstract_arch import AbsArchitecture


class Encoder(nn.Module):
    def __init__(self, in_features=96):
        super(Encoder, self).__init__()
        self.gconv1 = GCNConv(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.gconv2 = GCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.gconv3 = GCNConv(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward_stage(self, data, stage):
        assert (stage in ['gconv1', 'gconv2', 'gconv3', 'last'])

        if stage == 'gconv1':
            x, self.edge_index, self.batch = data.x, data.edge_index, data.batch
            x = self.gconv1(x, self.edge_index)
            x = self.bn1(x)
            x = self.relu(x)
            return x
        elif stage == 'gconv2':
            x = self.gconv2(data, self.edge_index)
            x = self.bn2(x)
            x = self.relu(x)
            return x
        elif stage == 'gconv3':
            x = self.gconv3(data, self.edge_index)
            x = self.bn3(x)
            x = self.relu(x)
            return x
        else:
            return data


class AttentionMask(nn.Module):
    def __init__(self, in_features):
        super(AttentionMask, self).__init__()
        self.features = in_features
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 2, kernel_size = 1)
        self.bn = nn.BatchNorm1d(2 * self.features)
        self.sigmod = nn.Sigmoid()

    def forward(self, data, encoderOuts):
        N = data.shape[0]
        x = data.view(N, 1, self.features)
        x = self.conv1(x)
        x = x.reshape(N, -1)
        x = self.bn(x)
        atta = self.sigmod(x)
        out = atta * encoderOuts
        return out
    

class AttentionMaskLast(nn.Module):
    def __init__(self, in_features, dropout=0.1):
        super(AttentionMaskLast, self).__init__()
        self.feature = in_features
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.bn1 = nn.BatchNorm1d(self.feature)
        self.bn2 = nn.BatchNorm1d(self.feature)
        self.relu = nn.ReLU()
        self.Linear = nn.Linear(self.feature, self.feature)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, data, batch):
        N = data.shape[0]
        x = data.view(N, 1, self.feature)
        x = self.conv1(x)
        x = x.reshape(N, -1)
        x = self.bn1(x)
        x = self.relu(x)
        x = global_max_pool(x, batch)
        x = self.Linear(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MTAN(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, 
                 stages = ['gconv1', 'gconv2', 'gconv3', 'last'], **kwargs):
        super(MTAN, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.stages = stages
        self.task_name = self.task_name
        self.encoder = encoder_class()
        self.attentionLayer1 = nn.ModuleDict({task: AttentionMask(in_features = 64) for task in self.task_name})
        self.attentionLayer2 = nn.ModuleDict({task: AttentionMask(in_features = 128) for task in self.task_name})
        self.attentionLayerLast = nn.ModuleDict({task: AttentionMaskLast(in_features = 256) for task in self.task_name})

    def forward(self, inputs, task_name):
        batch = inputs.batch
        for stage in self.stages:
            if stage == 'gconv1':
                self.inputs_origin = self.encoder.forward_stage(inputs, stage)
                inputs = self.inputs_origin
            else:
                inputs = self.encoder.forward_stage(inputs, stage)

            if stage == 'gconv2':
                self.AttentionCache = self.attentionLayer1[task_name](self.inputs_origin, inputs)
            elif stage == 'gconv3':
                self.AttentionCache = self.attentionLayer2[task_name](self.AttentionCache, inputs)
            elif stage == 'last':
                self.AttentionCache = self.attentionLayerLast[task_name](self.AttentionCache, batch)
            else:
                pass

        emb = {task_name:self.AttentionCache}
        out = {task_name:self.decoders[task_name](self.AttentionCache)}
        return out, emb

