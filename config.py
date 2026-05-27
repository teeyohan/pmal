import torch

def prepare_args(params):
    r"""Return the configuration of hyperparameters, optimizier, and learning rate scheduler.

    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    kwargs = {'weight_args': {}, 
              'arch_args': {}, 
              'sample_args': {}}

    if params.weighting in ['EW', 'UW', 'MGDA', 'GradNorm', 'DWA', 'PCGrad', 'CAGrad', 'GradVac', 'Align', 'Excess']:
        pass
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting)) 
        
    if params.arch in ['GCN', 'GAT', 'GIN', 'Cross_stitch', 'MTAN', 'MMoE', 'LTB', 'DSelect_k', 'ETR_NLP', 'Graphormer']:
        pass
    else:
        raise ValueError('No support architecture method {}'.format(params.arch)) 

    if params.sampling in ['Random', 'Entropy', 'K_Means', 'Core_set', 'BALD', 'LLoss', 'CDAL', 'BADGE', 'ProbCover', 'TiDAL', 'PMAL']:
        pass
    else:
        raise ValueError('No support sampling method {}'.format(params.sampling)) 
    
    if params.optim in ['adam']:
        if params.optim == 'adam':
            optim_param = {'optim': 'adam', 'lr': params.lr, 'weight_decay': params.weight_decay}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))
    
    display_args(params, kwargs, optim_param)
    
    return kwargs, optim_param

def display_args(params, kwargs, optim_param):

    print('='*40)
    print('General Configuration:')
    print('\tMode:', params.mode)
    print('\tWighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tSampling:', params.sampling)
    print('\tSave Path:', params.save_path)
    print('\tLoad Path:', params.load_path)
    print('\tDevice: {}'.format('cuda:'+params.gpu_id if torch.cuda.is_available() else 'cpu'))

    for wa, p in zip(['weight_args', 'arch_args', 'sample_args'], 
                     [params.weighting, params.arch, params.sampling]):
        if kwargs[wa] != {}:
            print('{} Configuration:'.format(p))
            for k, v in kwargs[wa].items():
                print('\t'+k+':', v)

    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)