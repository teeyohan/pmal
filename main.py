import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from trainer import Trainer
import weighting as weighting_method
import architecture as architecture_method
import sampling as sampling_method

from architecture.GCN import Encoder as Encoder_GCN
from architecture.GIN import Encoder as Encoder_GIN
from architecture.GAT import Encoder as Encoder_GAT
from architecture.Cross_stitch import Encoder as Encoder_Cross_stitch
from architecture.MTAN import Encoder as Encoder_MTAN
from architecture.MMoE import Encoder as Encoder_MMoE
from architecture.LTB import Encoder as Encoder_LTB
from architecture.DSelect_k import Encoder as Encoder_DSelect_k
from architecture.ETR_NLP import Encoder as Encoder_ETR_NLP
from architecture.Graphormer import Encoder as Encoder_Graphormer
from architecture.MDNHead import MDNHead

from metric import ClsMetric, RegMetric
from loss import BCELoss, MSELoss
from dataset import DataidxWrapper
from config import prepare_args

import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def main(params):
    kwargs, optim_param = prepare_args(params)

    task_name = [] # task
    n_init = {} # num_init
    n_query = {} # num_query
    if params.dataset in ['hiv', 'bace', 'bbbp', 'muv', 
                        'tox21', 'sider', 'clintox',
                        'qm7', 'esol', 'freesolv', 'lipophilicity',
                        'qm8', 'qm9', 'toxacute']:
        conf_path = os.path.join('data', params.dataset + '.txt')
        # switch on AL
        if params.switch == 'on':
            with open(conf_path, 'r') as file:
                for line in file.readlines():
                    [t_n, n_n, n_q] = line.strip().split(',')
                    task_name.append(t_n)
                    n_init[t_n] = float(n_n)
                    n_query[t_n] = float(n_q)
                    if (n_init[t_n] + params.n_round * n_query[t_n]) > 1.0:
                        raise ValueError('{}: the number of queries exceeds the total data'.format(t_n))
        # switch off AL
        elif params.switch == 'off':
            with open(conf_path, 'r') as file:
                for line in file.readlines():
                    [t_n, n_n, n_q] = line.strip().split(',')
                    task_name.append(t_n)
                    n_init[t_n] = 1.0
                    n_query[t_n] = 0.0
                    if (n_init[t_n] + params.n_round * n_query[t_n]) > 1.0:
                        raise ValueError('{}: the number of queries exceeds the total data'.format(t_n))
        else:
            raise ValueError('AL switch must be on/off!')
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))

    data_path = os.path.join('data', params.dataset + '.csv')
    # define cls tasks
    if params.dataset in ['hiv', 'bace', 'bbbp', 'muv', 
                          'tox21', 'sider', 'clintox']:
        task_type = 'cls'
        task_dict = {task: {'metrics': ['AUROC', 'AUPRC'],
                        'metrics_fn': ClsMetric(),
                        'loss_fn': BCELoss(),
                        'weight': [1, 1]} for task in task_name}
    # define reg tasks
    elif params.dataset in ['qm7', 'esol', 'freesolv', 'lipophilicity',
                            'qm8', 'qm9', 'toxacute']:
        task_type = 'reg'
        task_dict = {task: {'metrics': ['RMSE', 'R2'],
                        'metrics_fn': RegMetric(),
                        'loss_fn': MSELoss(),
                        'weight': [-1, 1]} for task in task_name}
    
    # prepare data
    data_idx_wrapper = DataidxWrapper(
        task_list=task_name, 
        data_path=data_path, 
        splitting=params.splitting, 
        valid_size=params.vs, 
        test_size=params.ts
        )
    dataset, data_idx = data_idx_wrapper.get_data_idx()

    n_total, n_pool, n_label, n_unlabel, n_val, n_test = {}, {}, {}, {}, {}, {}
    for task in task_name:
        n_pool[task] = len(data_idx[task]['pool']) # num_pool (label+unlabel)
        n_val[task] = len(data_idx[task]['val']) # num_val
        n_test[task] = len(data_idx[task]['test']) # num_test
        n_total[task] = n_pool[task] + n_val[task] + n_test[task] # num_total (label+unlabel+val+test)

        idx_lb = np.zeros(n_pool[task], dtype=bool)
        idx_tmp = np.arange(n_pool[task])
        np.random.shuffle(idx_tmp)
        n_init[task] = int(n_pool[task] * n_init[task])
        n_query[task] = int(n_pool[task] * n_query[task])
        idx_lb[idx_tmp[:n_init[task]]] = True

        data_idx[task]['label'] = [data_idx[task]['pool'][i] for i in np.where(idx_lb)[0]]
        data_idx[task]['unlabel'] = [data_idx[task]['pool'][i] for i in np.where(~idx_lb)[0]]

        n_label[task] = len(data_idx[task]['label']) # num_label
        n_unlabel[task] = len(data_idx[task]['unlabel']) # num_unlabel

        print('{} has {} samples\n\t{} Label, {} UnLabel, {} Query, {} Val, {} Test'.format(
            task, n_total[task], n_label[task], n_unlabel[task],
            n_query[task], n_val[task], n_test[task]))

    label_idx, unlabel_idx, val_idx, test_idx = {}, {}, {}, {}
    for task in task_name:
        label_idx[task] = data_idx[task]['label']
        unlabel_idx[task] = data_idx[task]['unlabel']
        val_idx[task] = data_idx[task]['val']
        test_idx[task] = data_idx[task]['test']
    
    # define encoders
    if params.arch == 'GCN':
        encoder = Encoder_GCN
    elif params.arch == 'GAT':
        encoder = Encoder_GAT
    elif params.arch == 'GIN':
        encoder = Encoder_GIN
    elif params.arch == 'Cross_stitch':
        encoder = Encoder_Cross_stitch
    elif params.arch == 'MTAN':
        encoder = Encoder_MTAN
    elif params.arch == 'MMoE':
        encoder = Encoder_MMoE
    elif params.arch == 'LTB':
        encoder = Encoder_LTB
    elif params.arch == 'DSelect_k':
        encoder = Encoder_DSelect_k
    elif params.arch == 'ETR_NLP':
        encoder = Encoder_ETR_NLP
    elif params.arch == 'Graphormer':
        encoder = Encoder_Graphormer
    else:
        raise ValueError('No support architecture method {}'.format(params.arch))
    
    # define decoders
    if params.switch == 'on' and params.sampling == 'PMAL':
        decoders = nn.ModuleDict({task: MDNHead(emb_size=256) for task in list(task_dict.keys())})
    else:
        decoders = nn.ModuleDict({task: nn.Linear(256, 1) for task in list(task_dict.keys())})

    # define model
    model = Trainer(task_dict=task_dict, 
                    weighting=weighting_method.__dict__[params.weighting], 
                    architecture=architecture_method.__dict__[params.arch], 
                    sampling=sampling_method.__dict__[params.sampling],
                    encoder_class=encoder, 
                    decoders=decoders,
                    optim_param=optim_param,
                    save_path=params.save_path,
                    load_path=params.load_path,
                    gpu_id=params.gpu_id,
                    **kwargs)
    
    # training
    if params.mode == 'train':
        # AL training
        if params.switch == 'on':
            print('='*40)
            print('AL training ...')
            for rd in range(1, params.n_round+1):
                print('='*40)
                print('AL Round {}'.format(rd))
                for task, idx in label_idx.items():
                    print('{}\thas {} Label, {} UnLabel, {} Query'.format(
                        task, len(label_idx[task]), len(unlabel_idx[task]), n_query[task]))
                label_idx, unlabel_idx = model.train(
                    task_type = task_type,
                    switch = params.switch,
                    sampling = params.sampling,
                    dataset = dataset,
                    n_total = n_total,
                    label_idx = label_idx, 
                    unlabel_idx = unlabel_idx,
                    val_idx = val_idx,
                    test_idx = test_idx, 
                    batch_size = params.bs,
                    epochs = params.epochs,
                    n_query = n_query,
                    )
        # general training
        if params.switch == 'off':
            print('='*40)
            print('General training ...')
            print('='*40)
            for task, idx in label_idx.items():
                print('{}\thas {} Label, {} UnLabel, {} Query'.format(
                    task, len(label_idx[task]), len(unlabel_idx[task]), n_query[task]))
            label_idx, unlabel_idx = model.train(
                task_type = task_type,
                switch = params.switch,
                sampling = params.sampling,
                dataset = dataset,
                n_total = n_total,
                label_idx = label_idx, 
                unlabel_idx = unlabel_idx,
                val_idx = val_idx,
                test_idx = test_idx, 
                batch_size = params.bs,
                epochs = params.epochs,
                n_query = n_query,
                )
    # testing
    elif params.mode == 'test':
        # model.test(dataset, test_idx)
        model.visualization(dataset, test_idx)
    else:
        raise ValueError


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Configurations')
    args.add_argument('--mode', type=str, default='test', help='train, test')
    args.add_argument('--gpu_id', default='0', type=str, help='gpu_id')
    args.add_argument('--save_path', type=str, default=None, help='save path')
    args.add_argument('--load_path', type=str, default='saved_model', help='load ckpt path')

    ## MTL weighting
    args.add_argument('--weighting', type=str, default='EW', help='MTL Weighting, option:\
                      +-------------------------------+\
                      |         MTL Weighting         |\
                      +-------------------------------+\
                      |            EW                 |\
                      |            UW                 |\
                      |            MGDA               |\
                      |            GradNorm           |\
                      |            DWA                |\
                      |            PCGrad             |\
                      |            CAGrad             |\
                      |            GradVac            |\
                      |            Align              |\
                      |            Excess             |\
                      +-------------------------------+\
                      ')
    ## MTL architecture
    args.add_argument('--arch', type=str, default='GCN', help='MTL Architecture, option:\
                      +-------------------------------+\
                      |        MTL Architecture       |\
                      +-------------------------------+\
                      |            GCN                |\
                      |            GAT                |\
                      |            GIN                |\
                      |            Cross_stitch       |\
                      |            MTAN               |\
                      |            MMoE               |\
                      |            LTB                |\
                      |            DSelect_k          |\
                      |            ETR_NLP            |\
                      |            Graphormer         |\
                      +-------------------------------+\
                      ')
    ## AL sampling
    args.add_argument('--switch', type=str, default='on', help='switch on/off AL training')
    args.add_argument('--sampling', type=str, default='PMAL', help='AL sampling, option:\
                      +-------------------------------+\
                      |           AL Sampling         |\
                      +-------------------------------+\
                      |            Random             |\
                      |            Entropy            |\
                      |            K_Means            |\
                      |            Core_set           |\
                      |            BALD               |\
                      |            LLoss              |\
                      |            CDAL               |\
                      |            BADGE              |\
                      |            ProbCover          |\
                      |            TiDAL              |\
                      |            PMAL               |\
                      +-------------------------------+\
                      ')
    args.add_argument('--n_round', type=int, default=1, help='AL round')

    ## optim
    args.add_argument('--optim', type=str, default='adam', help='optimizer')
    args.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')

    ## train
    args.add_argument('--dataset', type=str, default='toxacute', help='dataset, option:\
                      +---------------------------------------------------------------+\
                      |                     MoleculeNet+ DataSet                      |\
                      +-------------------------------+-------------------------------+\
                      |          Single-task          |           Multi-task          |\
                      +---------------+---------------+---------------+---------------+\
                      |      Cls      |      Reg      |      Cls      |      Reg      |\
                      +---------------+---------------+---------------+---------------+\
                      |      hiv      |      qm7      |      muv      |      qm8      |\
                      |      bace     |      esol     |     tox21     |      qm9      |\
                      |      bbbp     |    freesolv   |     sider     |    toxacute   |\
                      |               | lipophilicity |    clintox    |               |\
                      +---------------+---------------+-------------------------------+\
                      ')
    args.add_argument('--splitting', default='scaffold', type=str, help='splitting, option: random, scaffold')
    args.add_argument('--vs', default=0.1, type=float, help='valid size')
    args.add_argument('--ts', default=0.1, type=float, help='test size')
    args.add_argument('--bs', default=64, type=int, help='training batch size')
    args.add_argument('--epochs', default=30, type=int, help='training epochs')
    
    params = args.parse_args()
    main(params)