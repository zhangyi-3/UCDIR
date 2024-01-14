'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data

from functools import partial
import numpy as np
import random

from torch.utils.data import DataLoader

from utils.registry import DATASET_REGISTRY
from utils.dist_utils import get_dist_info
from data.data_sampler import EnlargedSampler


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    rank, ws = get_dist_info()
    sample_ratio = 1
    seed = 0
    # train_sampler = DATASET_REGISTRY.get('EnlargedSampler')(dataset, ws, rank, sample_ratio)
    train_sampler = EnlargedSampler(dataset, ws, rank, sample_ratio)

    if phase == 'train':
        train_loader = DataLoader(dataset, batch_size=dataset_opt['batch_size'], shuffle=False,
                                  num_workers=dataset_opt['num_workers'], pin_memory=True, sampler=train_sampler,
                                  drop_last=True, persistent_workers=True,
                                  worker_init_fn=partial(worker_init_fn, num_workers=dataset_opt['num_workers'],
                                                         rank=rank, seed=seed))
        # prefetcher = DATASET_REGISTRY.get('CUDAPrefetcher_consistent_batch')(train_loader)
        return train_loader
        # return torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=dataset_opt['batch_size'],
        #     shuffle=dataset_opt['use_shuffle'],
        #     num_workers=dataset_opt['num_workers'],
        #     pin_memory=True)
    elif phase == 'val':
        return DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=train_sampler,
        drop_last=False, persistent_workers=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    # print('datasetname' in dataset_opt.keys(), dataset_opt['datasetname'])
    if 'datasetname' in list(dataset_opt.keys()):
        import data.LRHR_dataset as proxy
        dataset = getattr(proxy, dataset_opt['datasetname'])(**dataset_opt['data_args'])
    else:
        from data.LRHR_dataset import LRHRDataset as D
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LR=(mode == 'LRHR')
                    )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
