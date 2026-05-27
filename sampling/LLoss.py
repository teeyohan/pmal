from sampling.abstract_sampling import AbsSampling
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LossNet(nn.Module):
    def __init__(self, task_name, feature_dim=256):
        """
        Args:
            feature_dim: 输入特征维度，与主模型输出的特征维度一致
            hidden_dim: 隐藏层维度
        """
        super(LossNet, self).__init__()
        
        self.decoders = nn.ModuleDict({task: nn.Linear(feature_dim, 1) for task in task_name})
        
    def forward(self, feature, task):
        """
        Args:
            feature: 主模型输出的特征 [batch_size, feature_dim]
        Returns:
            pred_loss: 预测的损失值 [batch_size]
        """
        pred_loss = self.decoders[task](feature)
        return pred_loss.view(-1) 
    
    def loss_forward(self, input, target, margin=1.0, reduction='mean'):
        if len(input) % 2 != 0: # the batch size is not 2B.
            input = input[:-1]
            target = target[:-1]
        # assert len(input) % 2 == 0, 'the batch size is not 2B.'
        assert input.shape == input.flip(0).shape
        
        input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)//2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
        
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0) # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()
        
        return loss
    

class LLoss(AbsSampling):
    def __init__(self):
        super(LLoss, self).__init__()

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        """使用预计算的uncertainty进行样本选择"""
        for task, idx in unlabel_idx.items():
            _unlabel_idx = self.list2array(n_total[task], unlabel_idx[task])
            uncertainty = unlabel_uncertainties[task][_unlabel_idx]  # 之前已经计算好的uncertainty
            
            selects = uncertainty.sort(descending=True)[1][:n_query[task]]
            queries = np.arange(n_total[task])[_unlabel_idx][selects].tolist()
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
            
        return label_idx, unlabel_idx