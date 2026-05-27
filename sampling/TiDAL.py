from sampling.abstract_sampling import AbsSampling
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TDNet(nn.Module):
    """用于TiDAL方法的网络,预测特征向量"""
    def __init__(self, task_name, feature_dim=256):
        
        super(TDNet, self).__init__()
        
        self.decoders = nn.ModuleDict({task: nn.Linear(feature_dim, feature_dim) for task in task_name})
        
    def forward(self, feature, task):
        """预测样本的特征向量
        Args:
            feature: 主模型输出特征 [batch_size, feature_dim]
        Returns:
            pred_feat: 预测的特征向量 [batch_size, feature_dim]
        """
        pred_feat = self.decoders[task](feature)
        return pred_feat
    

class TiDAL(AbsSampling):
    def __init__(self):
        super(TiDAL, self).__init__()

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        """使用预计算的uncertainty进行样本选择"""
        for task, idx in unlabel_idx.items():
            _unlabel_idx = self.list2array(n_total[task], unlabel_idx[task])
            # 直接使用已计算好的uncertainty
            uncertainty = unlabel_uncertainties[task][_unlabel_idx]
            
            # 选择不确定性最高的样本
            selects = uncertainty.sort(descending=True)[1][:n_query[task]]
            queries = np.arange(n_total[task])[_unlabel_idx][selects].tolist()
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
            
        return label_idx, unlabel_idx