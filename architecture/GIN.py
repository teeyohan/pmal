import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from architecture.abstract_arch import AbsArchitecture


class Encoder(nn.Module):
    def __init__(self, in_features=96, dropout=0.1, emb_size=256):
        super(Encoder, self).__init__()
        self.gconv1 = GINConv(nn.Linear(in_features, 64))
        self.bn1 = nn.BatchNorm1d(64)
        self.gconv2 = GINConv(nn.Linear(64, 128))
        self.bn2 = nn.BatchNorm1d(128)
        self.gconv3 = GINConv(nn.Linear(128, 256))
        self.bn3 = nn.BatchNorm1d(256)
        self.hidden = nn.Linear(256, emb_size)
        self.bn4 = nn.BatchNorm1d(emb_size)
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
        x = global_max_pool(x, batch)
        x = self.hidden(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    

class GIN(AbsArchitecture):
    r"""Hard Parameter Sharing (HPS-GIN).

    This method is proposed in `Multitask Learning: A Knowledge-Based Source of Inductive Bias (ICML 1993) <https://dl.acm.org/doi/10.5555/3091529.3091535>`_ \
    and implemented by us. 
    """
    def __init__(self, task_name, encoder_class, decoders, device, **kwargs):
        super(GIN, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.encoder = self.encoder_class()
