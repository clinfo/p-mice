#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from copy import deepcopy
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def normalize_cont(ar, ar_mean=None, ar_std=None, cont_index=None, eps=1e-10):
    '''
    Args:
        ar (2d-array):
        ar_mean (1d-array): 
        ar_std (1d-array):
        cont_index (1d-array):
    Returns:
        ar_norm (2d-array):
        ar_mean (1d-array):
        ar_std (1d-array):
    '''
    try:
        if cont_index is None:
            cont_index = [True] * ar.shape[1]
        # sanity check
        if ar.shape[1]!=len(cont_index):
            raise Exception('cont_index length inconsistence with array')

        if (ar_mean is None)&(ar_std is None): # for train
            ar_mean = ar[:, cont_index].mean(axis=0)
            ar_std = ar[:, cont_index].std(axis=0)
        else: # for test
            # sanity check
            if (len(ar_mean)!=sum(cont_index))|(len(ar_std)!=sum(cont_index)):
                raise Exception('Number of continuous variable was inconsistent')

        ar_norm = deepcopy(ar)
        ar_norm[:, cont_index] = (ar[:, cont_index] - ar_mean) / (ar_std + eps)
    except Exception as e:
        logging.exception(f'{e}')
        ar_norm, ar_mean, ar_std = None, None, None
    return ar_norm, ar_mean, ar_std

def denormalize_cont(ar, ar_mean, ar_std, cont_index=None):
    '''
    Args:
        ar (2d-array):
        ar_mean (1d-array): 
        ar_std (1d-array):
        cont_index (1d-array):
    Returns:
        ar_denorm (2d-array): 
    '''
    try:
        if cont_index is None:
            cont_index = [True] * ar.shape[1]
        # sanity check
        if ar.shape[1]!=len(cont_index):
            raise Exception('cont_index length inconsistence with array')
        if (ar_mean is None)|(ar_std is None):
            raise Exception('ar_mean or ar_std is undefined')
        if (len(ar_mean)!=sum(cont_index))|(len(ar_std)!=sum(cont_index)):
            raise Exception('Number of continuous variable was inconsistent')
        
        ar_denorm = deepcopy(ar)
        ar_denorm[:, cont_index] = (ar[:, cont_index] * ar_std) + ar_mean
    except Exception as e:
        logging.exception(f'{e}')
        ar_denorm = None
    return ar_denorm
        