from sampling.abstract_sampling import AbsSampling
import torch
import numpy as np
import pandas as pd


DELTA = 0.1
BATCH_SIZE = 500

class ProbCover(AbsSampling):
    def __init__(self):
        super(ProbCover, self).__init__()

    def construct_graph(self, features, batch_size: int = BATCH_SIZE):
        """
        分批构建稀疏相似图，避免一次性计算NxN距离
        x → y  当且仅当  L2(x, y) < DELTA
        """
        device = getattr(self, 'device', 'cpu')
        feats = torch.as_tensor(features, device=device)

        xs, ys, ds = [], [], []
        n = feats.size(0)
        for start in range(0, n, batch_size):
            cur = feats[start:start + batch_size]
            dist = torch.cdist(cur, feats)        
            mask = dist < DELTA
            x, y = mask.nonzero(as_tuple=True)
            xs.append(x.cpu() + start)          
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()
        return pd.DataFrame({'x': xs, 'y': ys, 'd': ds})

    def select_samples(self, label_idx, unlabel_idx, label_embds, unlabel_embds, n_query):
        label_idx_array = np.array(label_idx)
        unlabel_idx_array = np.array(unlabel_idx)
        relevant_indices = np.concatenate([label_idx_array, unlabel_idx_array]).astype(int)

        label_embds_array = np.array(label_embds)[label_idx_array, :]
        unlabel_embds_array = np.array(unlabel_embds)[unlabel_idx_array, :]
        features = np.concatenate([label_embds_array, unlabel_embds_array], axis=0)
        
        graph_df = self.construct_graph(features)

        edge_from_seen = np.isin(graph_df.x, np.arange(len(label_idx_array)))
        covered_samples = graph_df.y[edge_from_seen].unique()
        cur_df = graph_df[(~np.isin(graph_df.y, covered_samples))]

        selected = []
        for i in range(n_query):
            degrees = np.bincount(cur_df.x, minlength=len(relevant_indices))  # 这里的前117都被去掉了，说明只包含未标注样本
            degrees[:len(label_idx_array)] = -1   # 屏蔽已标记节点
            if selected:
                degrees[selected] = -1              # 屏蔽已选过节点
            cur = degrees.argmax()

            # 若全部度数 ≤ 0，则随机补选未标记但尚未被选过的节点
            if degrees[cur] < 0:
                remain = np.setdiff1d(
                    np.arange(len(label_idx_array), len(relevant_indices)),
                    selected, assume_unique=False
                )
                if len(remain) == 0:
                    break
                cur = np.random.choice(remain)

            # 选择当前节点
            selected.append(cur)
            
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            # cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]
            # --------- 修改 ①：同时去掉以 cur 为起点的边 ---------
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples)) & (cur_df.x != cur)]

            # covered_samples = np.concatenate([covered_samples, new_covered_samples]) #?

        activeSet = relevant_indices[selected].tolist()
        activeSet = list(dict.fromkeys(activeSet))   # 保持原顺序并去重
        # print(f"min selected position: {min(selected)}, max selected position: {max(selected)}, n_labeled: {len(label_idx_array)}")
        return activeSet
    

    def query(self, n_total, n_query, label_idx, unlabel_idx, label_embds, unlabel_embds, unlabel_drops, unlabel_uncertainties):
        """
        在多任务学习中选择样本：
        每次选择指定数量的样本，并更新索引。
        """
        for task, idx in unlabel_idx.items():
            # 使用 ProbCover 选择样本
            queries = self.select_samples(label_idx[task], unlabel_idx[task], label_embds[task], unlabel_embds[task], n_query[task])
            label_idx[task], unlabel_idx[task] = self.update_idx(queries, label_idx[task], unlabel_idx[task])

        return label_idx, unlabel_idx

