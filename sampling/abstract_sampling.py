import numpy as np

class AbsSampling():
    r"""
    """
    def __init__(self):
        super(AbsSampling, self).__init__()
    
    def list2array(self, n, idx):
        boolarray = np.zeros(n, dtype=bool)
        boolarray[idx] = True
        return boolarray
    
    def update_idx(self, queries, label_idx, unlabel_idx):
        for element in queries:
            label_idx.append(element) 
            unlabel_idx.remove(element)
        return label_idx, unlabel_idx
    
    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        r"""
        """
        return label_idx, unlabel_idx