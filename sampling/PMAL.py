from sampling.abstract_sampling import AbsSampling
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PMAL(AbsSampling):
    def __init__(self):
        super(PMAL, self).__init__()

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        """使用预计算的uncertainty进行样本选择"""
        for task, idx in unlabel_idx.items():
            _unlabel_idx = self.list2array(n_total[task], unlabel_idx[task])
            uncertainty = unlabel_uncertainties[task][_unlabel_idx]  # 之前已经计算好的uncertainty
            
            selects = uncertainty.sort(descending=True)[1][:n_query[task]]
            queries = np.arange(n_total[task])[_unlabel_idx][selects].tolist()
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
            
        return label_idx, unlabel_idx