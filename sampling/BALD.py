from sampling.abstract_sampling import AbsSampling
import torch
import torch.nn.functional as F
import numpy as np


class BALD(AbsSampling):
    def __init__(self):
        super(BALD, self).__init__()

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        for task, idx in unlabel_idx.items():
            _unlabel_idx = self.list2array(n_total[task], unlabel_idx[task])
            _unlabel_drops = unlabel_drops[task][:, _unlabel_idx, :]
            unlabel_drop_probs = F.softmax(_unlabel_drops, dim=2)
            pb = unlabel_drop_probs.mean(0)
            A = (-pb*torch.log(pb)).sum(1)
            B = (-unlabel_drop_probs*torch.log(unlabel_drop_probs)).sum(2).mean(0)
            U = B - A
            selects = U.sort()[1][:n_query[task]]
            queries = np.arange(n_total[task])[_unlabel_idx][selects].tolist()
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
        return label_idx, unlabel_idx