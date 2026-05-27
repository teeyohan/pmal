from sampling.abstract_sampling import AbsSampling
from sklearn.cluster import KMeans
import numpy as np


class K_Means(AbsSampling):
    def __init__(self):
        super(K_Means, self).__init__()

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        for task, idx in unlabel_idx.items():
            _unlabel_idx = self.list2array(n_total[task], unlabel_idx[task])
            _unlabel_embds = unlabel_embds[task][_unlabel_idx, :].numpy()

            cluster_learner = KMeans(n_clusters=n_query[task])
            cluster_learner.fit(_unlabel_embds)
            cluster_idxs = cluster_learner.predict(_unlabel_embds)
            centers = cluster_learner.cluster_centers_[cluster_idxs]
            dis = (_unlabel_embds - centers)**2
            dis = dis.sum(axis=1)
            
            selects = np.array([np.arange(_unlabel_embds.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n_query[task])])
            queries = np.arange(n_total[task])[_unlabel_idx][selects].tolist()
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
        return label_idx, unlabel_idx