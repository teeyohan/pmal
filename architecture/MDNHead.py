import torch
import torch.nn as nn
import torch.nn.functional as F 


class MDNHead(nn.Module):
    def __init__(self, K=4, emb_size=256):
        super(MDNHead, self).__init__()
        self.K = K
        self.fc_pi = nn.Linear(emb_size, K)
        self.fc_mu = nn.Linear(emb_size, K)
        self.fc_var = nn.Linear(emb_size, K)

    def forward(self, x):
        pi = self.fc_pi(x)  # [batch_size, k]
        pi = F.softmax(pi, dim=-1)

        mu = self.fc_mu(x)  # [batch_size, k]
        mu = mu.view(-1, self.K, 1)

        var = self.fc_var(x)  # [batch_size, k]
        var = var.view(-1, self.K, 1)
        var = F.softplus(var)
        
        return pi, mu, var