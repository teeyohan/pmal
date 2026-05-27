import csv
import numpy as np

import torch
from torch_geometric.data import Data, Dataset

import networkx as nx

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def read_smiles(data_path, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            # if i != 0:
            smiles = row['smiles']
            label = row[task]
            edges = []
            mol = Chem.MolFromSmiles(smiles)
            if mol != None:
                for bond in mol.GetBonds():
                    edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            if (mol != None) and (edges != []) and (label != ''):
                smiles_data.append(smiles)
                labels.append(float(label))
        # print(len(smiles_data))
    return smiles_data, labels

def atom_features(atom):
    fea_Symbol = one_of_k_encoding_unk(atom.GetSymbol(),
                                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                    'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                    'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                    'Pt', 'Hg', 'Pb', 'Unknown'])
    fea_Degree = one_of_k_encoding_unk(atom.GetDegree(), list(range(17)))
    fea_TotalNumHs = one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(17)))
    fea_ImplicitValence = one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(17)))
    fea_IsAromatic = [atom.GetIsAromatic()]
    return np.array(fea_Symbol + fea_Degree + fea_TotalNumHs + fea_ImplicitValence + fea_IsAromatic)

# def one_of_k_encoding(x, allowable_set):
#     if x not in allowable_set:
#         raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
#     return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def random_split(dataset, task, valid_size, test_size):
    data_len = len(dataset)
    # print("%s has %d samples" % (task, data_len))
    indices = list(range(data_len))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * data_len))
    split2 = int(np.floor(test_size * data_len))
    valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
    return train_idx, valid_idx, test_idx

def scaffold_balanced_split(dataset, task, valid_size, test_size):
    from collections import defaultdict
    
    labels = dataset.labels
    total_samples = len(dataset)
    scaffolds_dict = defaultdict(list)
    for idx, smiles in enumerate(dataset.smiles_data):
        scaffold = _generate_scaffold(smiles)
        scaffolds_dict[scaffold].append(idx)
    
    sorted_indices = []
    for scaffold, indices in scaffolds_dict.items():
        sorted_within_scaffold = sorted(indices, key=lambda idx: labels[idx])
        sorted_indices.extend(sorted_within_scaffold)
    
    train_target = int((1.0 - valid_size - test_size) * total_samples)
    valid_target = int(valid_size * total_samples)
    test_target = total_samples - train_target - valid_target
    
    train_inds = sorted_indices[:train_target]
    valid_inds = sorted_indices[train_target:train_target + valid_target]
    test_inds = sorted_indices[train_target + valid_target:]
    
    return train_inds, valid_inds, test_inds

def scaffold_split(dataset, task, valid_size, test_size):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset, task)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)

    train_inds = []
    valid_inds = []
    test_inds = []
    # print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

def generate_scaffolds(dataset, task):
    scaffolds = {}
    data_len = len(dataset)
    # print("%s has %d samples" % (task, data_len))

    # print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

class DatasetWrapper(Dataset):
    def __init__(self, data_path, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, task)
        # 初始化 moving_probs，用于 TiDAL
        self.moving_probs = torch.zeros(size=(len(self.labels), 256), dtype=torch.float)
        
    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature)
            # features.append(feature/sum(feature))

        edges, edge_index = [], []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        g = nx.Graph(edges).to_directed()
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        data = Data(x = torch.FloatTensor(features),
                    edge_index = torch.LongTensor(edge_index).transpose(1, 0),
                    y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1))

        return data, index

    def __len__(self):
        return len(self.labels)
    
class DataidxWrapper():
    def __init__(self, task_list, data_path, splitting, valid_size, test_size):
        super().__init__()
        self.task_list = task_list
        self.data_path = data_path
        self.splitting = splitting
        self.valid_size = valid_size
        self.test_size = test_size
        assert splitting in ['random', 'scaffold']

    def get_data_idx(self):
        dataset, data_idx = {}, {}
        print("=" * 40)
        for k, task in enumerate(self.task_list):
            data_idx[task] = {}
            dataset[task] = DatasetWrapper(self.data_path, task)
            data_idx[task]['pool'], data_idx[task]['val'], data_idx[task]['test'] =\
                self.get_train_val_data_idx(dataset[task], task)
        return dataset, data_idx

    def get_train_val_data_idx(self, dataset, task):
        if self.splitting == 'random':
            pool_idx, valid_idx, test_idx = random_split(dataset, task, self.valid_size, self.test_size)
        elif self.splitting == 'scaffold':
            if 'MUV' in task: # MUV is extremely unlalanced
                pool_idx, valid_idx, test_idx = scaffold_balanced_split(dataset, task, self.valid_size, self.test_size)
            else:
                pool_idx, valid_idx, test_idx = scaffold_split(dataset, task, self.valid_size, self.test_size)
        return pool_idx, valid_idx, test_idx