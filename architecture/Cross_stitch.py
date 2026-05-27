import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from architecture.abstract_arch import AbsArchitecture


class Encoder(nn.Module):
    def __init__(self, in_features=96, dropout=0.1, emb_size=256):
        super(Encoder, self).__init__()
        self.gconv1 = GCNConv(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.gconv2 = GCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.gconv3 = GCNConv(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.hidden = nn.Linear(256, emb_size)
        self.bn4 = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward_stage(self, data, stage):
        assert (stage in ['gconv1', 'gconv2', 'gconv3', 'hidden', 'last'])

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
        elif stage == 'hidden':
            x = global_max_pool(data, self.batch)
            x = self.hidden(x)
            x = self.bn4(x)
            x = self.relu(x)
            return x
        else:
            x = self.dropout(data)
            return x
        

class ChannelWiseMultiply(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWiseMultiply, self).__init__()
        self.param = nn.Parameter(torch.FloatTensor(num_channels), requires_grad=True)

    def init_value(self, value):
        with torch.no_grad():
            self.param.data.fill_(value)

    def forward(self, x):
        return torch.mul(self.param.view(1, -1), x)


class CrossStitchUnit(nn.Module):
    def __init__(self, tasks, num_channels, alpha, beta):
        super(CrossStitchUnit, self).__init__()
        self.cross_stitch_unit = nn.ModuleDict(
            {t: nn.ModuleDict({t: ChannelWiseMultiply(num_channels) for t in tasks}) 
             for t in tasks})
        for t_i in tasks:
            for t_j in tasks:
                if t_i == t_j:
                    self.cross_stitch_unit[t_i][t_j].init_value(alpha)
                else:
                    self.cross_stitch_unit[t_i][t_j].init_value(beta)

    def forward(self, task_features):
        out = {}
        for t_i in task_features.keys():
            prod = torch.stack([self.cross_stitch_unit[t_i][t_j](task_features[t_j]) for t_j in task_features.keys()])
            out[t_i] = torch.sum(prod, dim=0)
        return out


class Cross_stitch(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, 
                 stages=['gconv1', 'gconv2', 'gconv3', 'hidden', 'last'],
                 stagesUnit=['gconv1', 'gconv2', 'gconv3', 'hidden'],
                 channels={'gconv1': 64, 'gconv2': 128, 'gconv3': 256, 'hidden': 256}, 
                 alpha=0.9, beta=0.1, **kwargs):
        
        super(Cross_stitch, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.stages = stages
        self.stagesUnit = stagesUnit
        self.channels = channels
        self.task_name = task_name
        self.encoder = nn.ModuleDict({task: self.encoder_class() for task in self.task_name})

        self.cross_stitch = nn.ModuleDict(
            {stage: CrossStitchUnit(self.task_name, channels[stage], alpha, beta) 
             for stage in self.stagesUnit}
             )

    def forward(self, inputs, task_name = None):
        inputs = {task: inputs for task in self.task_name}

        for stage in self.stages:
            for task in self.task_name:
                inputs[task] = self.encoder[task].forward_stage(inputs[task], stage)
            if stage != 'last':
                inputs = self.cross_stitch[stage](inputs)

        emb = {task: inputs[task] for task in self.task_name}
        out = {task: self.decoders[task](inputs[task]) for task in self.task_name}
        return out, emb
