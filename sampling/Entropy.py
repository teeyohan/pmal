from sampling.abstract_sampling import AbsSampling
import torch
import torch.nn.functional as F
import numpy as np


class Entropy(AbsSampling):
    def __init__(self):
        super(Entropy, self).__init__()

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        for task, idx in unlabel_idx.items():
            _unlabel_idx = self.list2array(n_total[task], unlabel_idx[task])
            _unlabel_embds = unlabel_embds[task][_unlabel_idx, :]
            unlabel_embd_probs = F.softmax(_unlabel_embds, dim=1)
            log_unlabel_embd_probs = torch.log(unlabel_embd_probs)
            U = (unlabel_embd_probs*log_unlabel_embd_probs).sum(1)
            selects = U.sort()[1][:n_query[task]]
            queries = np.arange(n_total[task])[_unlabel_idx][selects].tolist()
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
        return label_idx, unlabel_idx