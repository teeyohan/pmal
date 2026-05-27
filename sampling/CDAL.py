from sampling.abstract_sampling import AbsSampling
import torch.nn.functional as F
import numpy as np
import torch

class CDAL(AbsSampling):
    def __init__(self):
        super(CDAL, self).__init__()

    def select_coreset(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = self.pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)
        idxs = []
        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = self.pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
            for idx in idxs:
                min_dist[idx] = -np.inf
        return idxs

    def pairwise_distances(self, a, b):
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        dist = np.zeros((a.size(0), b.size(0)), dtype=np.float64)
        for i in range(a.size(0)):
            for j in range(b.size(0)):
                dist[i][j] = self.KL_symmetric(a[i], b[j])
        return np.array(dist)

    def KL_symmetric(self, a, b):
        kl1 = a * torch.log(a / b)
        kl2 = b * torch.log(b / a)
        kl = 0.5 * (torch.sum(kl1)) + 0.5 * (torch.sum(kl2))
        return kl

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        for task, idx in unlabel_idx.items():
            A = np.array(F.softmax(unlabel_embds[task][unlabel_idx[task], :], dim=1))
            B = np.array(F.softmax(label_embds[task][label_idx[task], :], dim=1))
            selects = self.select_coreset(A, B, n_query[task])
            queries = [unlabel_idx[task][i] for i in selects]
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
        return label_idx, unlabel_idx