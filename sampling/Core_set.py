from sampling.abstract_sampling import AbsSampling
import numpy as np
from sklearn.metrics import pairwise_distances


class Core_set(AbsSampling):
    def __init__(self):
        super(Core_set, self).__init__()

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)
        idxs = []
        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
            for idx in idxs:
                min_dist[idx] = -np.inf
        return idxs

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        for task, idx in unlabel_idx.items():
            A = np.array(unlabel_embds[task])[unlabel_idx[task], :]
            B = np.array(label_embds[task])[label_idx[task], :]
            selects = self.furthest_first(A, B, n_query[task])
            queries = [unlabel_idx[task][i] for i in selects]
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
        return label_idx, unlabel_idx