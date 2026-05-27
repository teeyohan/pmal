from sampling.abstract_sampling import AbsSampling
from random import sample


class Random(AbsSampling):
    def __init__(self):
        super(Random, self).__init__()
        
    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        for task, idx in unlabel_idx.items():
            queries = sample(unlabel_idx[task], n_query[task])
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])
        return label_idx, unlabel_idx