import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool

from architecture.abstract_arch import AbsArchitecture


class Encoder(nn.Module):
    def __init__(self, in_features=96, out_features=256):
        super(Encoder, self).__init__()
        self.gconv1 = GCNConv(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.gconv2 = GCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.gconv3 = GCNConv(128, out_features)
        self.bn3 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

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
        return x

class Expert(nn.Module):
    def __init__(self, in_features, dropout, out_features):
        super(Expert, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data, batch):
        x = global_max_pool(data, batch)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Logit(nn.Module):
    def __init__(self, in_features, dropout, out_features):
        super(Logit, self).__init__()
        self.gconv = GCNConv(in_channels = in_features, out_channels = 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gconv(x, edge_index)
        x = self.bn1(x)
        x = global_max_pool(x, batch)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    
class DSelect_k(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, num_experts=4, input_features = 96, **kwargs):
        super(DSelect_k, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.encoder = encoder_class(in_features=96, out_features=256)
        self.experts = nn.ModuleList(
            [Expert(in_features = 256, dropout=0.1, out_features = 256) for _ in range(num_experts)]
            )
        self.num_experts = num_experts
        # input_dim = self.encoder.gconv1.in_channels
        # self.gates = nn.ModuleDict(
        #     {task: Gate(in_features = input_dim, num_experts = num_experts) for task in self.task_name}
        #     )
        ## DSelect_k unique
        self._num_nonzeros = 2 # definitely features
        self._gamma = 1
        self._num_binary = math.ceil(math.log2(num_experts))
        self._power_of_2 = (num_experts == 2 ** self._num_binary)
        self._z_logits = nn.ModuleDict({task:Logit(
            in_features=input_features, dropout=0.1, out_features=self._num_nonzeros*self._num_binary) for task in self.task_name})
        self._w_logits = nn.ModuleDict({task:Logit(
            in_features=input_features, dropout=0.1, out_features=self._num_nonzeros) for task in self.task_name})
        # initialization
        for param in self._z_logits.parameters():
            param.data.uniform_(-self._gamma/100, self._gamma/100)
        for param in self._w_logits.parameters():
            param.data.uniform_(-0.05, 0.05)

        binary_matrix = np.array([list(np.binary_repr(val, width = self._num_binary)) \
                                  for val in range(self.num_experts)]).astype(bool)
        self._binary_codes = torch.from_numpy(binary_matrix).to(self.device).unsqueeze(0)

        self.gate_specific = None

    def _smooth_step_fun(self, t, gamma = 1.0):
        return torch.where(t <= -gamma / 2, torch.zeros_like(t, device = t.device),
                           torch.where(t >= gamma / 2, torch.ones_like(t, device = t.device),
                                       (-2 / (gamma ** 3)) * (t ** 3) + (3 / (2 * gamma)) * t + 1 / 2))

    def _entropy_reg_loss(self, inputs):
        loss = -(inputs * torch.log(inputs + 1e-6)).sum() * 1e-6
        if not self._power_of_2:
            loss += (1 / inputs.sum(-1)).sum()
        loss.backward(retain_graph = True)

    def forward(self, data, task_name=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data_public = self.encoder(data)
        # weights = self.gates[task_name](data)
        experts_outputs = torch.stack([expert(data_public, batch) for expert in self.experts])
        sample_logits = self._z_logits[task_name](data)
        sample_logits = sample_logits.reshape(-1, self._num_nonzeros,1, self._num_binary)
        smooth_step_activations = self._smooth_step_fun(sample_logits)
        weights_out = torch.where(self._binary_codes.unsqueeze(0), smooth_step_activations, 
                                  1 - smooth_step_activations).prod(3)
        weight_selector = F.softmax(self._w_logits[task_name](data), dim = 1)
        experts_weights = torch.einsum('ij, ij... -> i...', weight_selector, weights_out)
        # torch.einsum() modify the shape of weight suiting outputs
        # weights = weights.unsqueeze(1).permute(2, 0, 1)
        # weights = torch.mean(weights).reshape((-1, 1, 1))
        outputs = torch.einsum('ij...,ji->j...',  experts_outputs, experts_weights)

        emb = {task: outputs for task in self.task_name}
        out = {task: self.decoders[task](outputs) for task in self.task_name}
        return out, emb