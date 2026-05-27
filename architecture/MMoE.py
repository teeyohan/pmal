import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from architecture.abstract_arch import AbsArchitecture


class Encoder(nn.Module):
    def __init__(self, in_features=96, dropout=0.1, out_features=256):
        super(Encoder, self).__init__()
        self.gconv1 = GCNConv(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.gconv2 = GCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.gconv3 = GCNConv(128, out_features)
        self.bn3 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gconv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gconv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.gconv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Expert(nn.Module):
    def __init__(self, in_features , out_features):
        super(Expert, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, data, batch):
        x = global_max_pool(data, batch)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Gate(nn.Module):
    def __init__(self, num_experts, in_features = 78):
        super(Gate, self).__init__()
        self.gconv = GCNConv(in_channels = in_features, out_channels = 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, num_experts)
        self.bn2 = nn.BatchNorm1d(num_experts)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gconv(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.softmax(x)
        return x
    
class MMoE(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, num_experts=4, **kwargs):
        super(MMoE, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.encoder = encoder_class(in_features=96, dropout=0.1, out_features=256)
        self.experts = nn.ModuleList(
            [Expert(in_features = 256, out_features = 256) for _ in range(num_experts)]
            )
        input_dim = self.encoder.gconv1.in_channels
        self.gates = nn.ModuleDict(
            {task: Gate(in_features = input_dim, num_experts = num_experts) for task in self.task_name}
            )

    def forward(self, data, task_name=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data_public = self.encoder(data)
        weights = self.gates[task_name](data)
        experts_outputs = torch.stack([expert(data_public, batch) for expert in self.experts])
        # torch.einsum() modify the shape of weight suiting outputs
        # weights = weights.unsqueeze(1).permute(2, 0, 1)
        # weights = torch.mean(weights).reshape((-1, 1, 1))
        outputs = torch.einsum('ij...,ji->j...', experts_outputs, weights)

        emb = {task: outputs for task in self.task_name}
        out = {task: self.decoders[task](outputs) for task in self.task_name}
        return out, emb