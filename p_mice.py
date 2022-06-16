#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import logging
import numpy as np

from sklearn.utils.extmath import cartesian
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as patches
import seaborn as sns

import utils

class PdcParams:
    '''
    model (model):
    ar_feature_names (1d-array[str]): 
    intervention_variables (list[str]):
    threshold (float): default=0.5
    roughness (float): default=0.05, should be range of [0, 1]
    cont_index (1d-array[bool]): default=None  
    ar_p_min (float): default=None
    ar_p_max (float): default=None
    ar_step (1d-array):
    iv_index (list[int]): iv_index for ar_feature_names
    original_record (2d-array):
    ar_original_iv_values (1d-array):
    original_predict_proba (float):
    original_pred (int):
    list_iv_values (list[1d-array])
    ar_iv_perturbed_values (2d-array)
    original_idx (1d-array[int])
    ar_label_proba (1d-array)
    ar_label (1d-array[int])
    ar_searched_index (1d-array[int])
    ar_unsearched_index (1d-array[int])
    '''
    def __init__(self,
                 model,
                 ar_feature_names, 
                 intervention_variables,
                 threshold=0.5,
                 roughness=0.05,
                 cont_index=None):
        '''
        Args:
            model (model):
            ar_feature_names (1d-array[str]): 
            intervention_variables (list[str]):
            threshold (float): default=0.5
            roughness (float): should be range of [0, 1]
            cont_index (1d-array[bool]): default=None
        '''
        self.model = model
        self.ar_feature_names = np.array(ar_feature_names)
        self.intervention_variables = list(intervention_variables)
        self.threshold = threshold
        self.roughness = roughness
        if cont_index is None:
            self.ar_cont_index = np.array([True]*len(self.ar_feature_names))
        else:
            self.ar_cont_index = np.array(cont_index)
        iv_index = []
        for iv in intervention_variables:
            iv_index.append(np.where(ar_feature_names==iv)[0][0])
        self.iv_index = iv_index
        ar_iv_index_mask = np.zeros(len(ar_feature_names), dtype=bool)
        ar_iv_index_mask[iv_index] = True
        ar_niv_index_mask = np.logical_not(ar_iv_index_mask)
        self.ar_iv_index_mask = ar_iv_index_mask
        self.ar_niv_index_mask = ar_niv_index_mask
        # iv_range: set later
        self.ar_p_min = None
        self.ar_p_max = None
        # step: set later
        self.ar_step = None
        # original values: set later
        self.original_record = None
        self.ar_original_iv_values = None
        self.original_predict_proba = None
        self.original_pred = None
        # iv perturbation
        self.list_iv_values = None
        self.ar_iv_perturbed_values = None
        self.original_idx = None
        self.ar_label_proba = None
        self.ar_label = None
        self.ar_searched_index = None
        self.ar_unsearched_index = None
        
    def set_range(self, ar_ref_X=None, ar_p_min=None, ar_p_max=None):
        '''
        Args:
            ar_ref_X (2d-array):
            ar_p_min (1d-array):
            ar_p_max (1d-array):
        '''
        if ar_p_min is None:
            if ar_ref_X is None: # sanity check
                raise Exception('Neither ar_p_min or ar_ref_X was stated')
            self.ar_p_min = ar_ref_X[:, self.iv_index].min(axis=0)
        else:
            self.ar_p_min = ar_p_min
        if ar_p_max is None:
            if ar_ref_X is None: # sanity check
                raise Exception('Neither ar_p_min or ar_ref_X was stated')
            self.ar_p_max = ar_ref_X[:, self.iv_index].max(axis=0)
        else:
            self.ar_p_max = ar_p_max
            
        # set step
        ar_range = self.ar_p_max - self.ar_p_min
        self.ar_step = ar_range * self.roughness
        # logging.debug('')
                
    def set_record(self, record):
        '''
        overwrite
        Args:
            record (2d-array):    
        '''
        # set
        self.original_record = record
        self.ar_original_iv_values = record[0, self.iv_index]
        self.original_predict_proba = self.model.predict_proba(record)[0][1]
        self.original_pred = 1 if self.original_predict_proba>=self.threshold else 0
        # logging.debug('')
            
    def calc_perturbed_values(self):
        '''
        ar_p_min, ar_p_max, ar_step, ar_original_iv_values must be set.
        overwrite
        '''
        list_iv_values = []
        for i, iv_i in enumerate(self.iv_index):
            # initialize
            original_iv_value = self.ar_original_iv_values[i]
            step = self.ar_step[i]
            p_min = self.ar_p_min[i]
            p_max = self.ar_p_max[i]                
            list_iv_values_tmp = [original_iv_value]
            
            iv_tmp = original_iv_value - step
            while iv_tmp>=p_min:
                list_iv_values_tmp.insert(0, iv_tmp)
                iv_tmp -= step
            iv_tmp = original_iv_value + step
            while iv_tmp<=p_max:
                list_iv_values_tmp.append(iv_tmp)
                iv_tmp += step            
                
            list_iv_values.append(np.array(list_iv_values_tmp))
            
        # combination     
        ar_iv_perturbed_values = cartesian(list_iv_values)
        ar_label_proba = np.repeat(-1., ar_iv_perturbed_values.shape[0])
        ar_label = np.repeat(-1, ar_iv_perturbed_values.shape[0])
        original_idx = np.where(np.all(np.isclose(ar_iv_perturbed_values, self.ar_original_iv_values), axis=1))[0] # np.array
        # sanity check
        if len(original_idx)!=1:
            raise Exception('Error in original_idx')
        
        ar_label_proba[original_idx] = self.original_predict_proba
        ar_label[original_idx] = self.original_pred
        
        # set
        self.list_iv_values = list_iv_values # list[1d-array]
        self.ar_iv_perturbed_values = ar_iv_perturbed_values
        self.original_idx = original_idx
        self.ar_label_proba = ar_label_proba
        self.ar_label = ar_label
        self.ar_searched_index = original_idx
        self.ar_unsearched_index = np.delete(np.arange(ar_label.shape[0]), original_idx)
        # logging.debug('')
    
    def make_X_ice_data(self):
        '''
        Full search
        Return:
            ar_X_ice (2d-array):
        '''
        ar_X_ice = deepcopy(np.repeat(self.original_record, self.ar_unsearched_index.shape[0], axis=0))
        ar_X_ice[:, self.iv_index] = self.ar_iv_perturbed_values[self.ar_unsearched_index]
        # logging.debug('')
        return ar_X_ice

    def set_pred(self, ar_X_ice):
        '''
        Full search
        Args:
            ar_X_ice (2d-array):
        Return:
        '''
        ar_pred_proba = self.model.predict_proba(ar_X_ice)[:, 1]
        ar_pred = np.where(ar_pred_proba>=self.threshold, 1, 0)
        # update
        self.ar_label_proba[self.ar_unsearched_index] = ar_pred_proba
        self.ar_label[self.ar_unsearched_index] = ar_pred
        self.ar_searched_index = np.append(self.ar_searched_index, self.ar_unsearched_index)
        self.ar_unsearched_index = np.where(self.ar_label==-1)[0]
        # logging.debug('')
    
    def make_ref_data_for_knn(self, ar_ref_X, same_label, ar_ref_y=None):
        '''
        1. if same_label: slice
        2. add original_record
        3. normalized
        Args:
            ar_ref_X (2d-array):
            same_label (bool):
            ar_ref_y (1d-array[int]): if same_label, must be defined.
        '''
        # normalization # [TODO] Only continuous variables are supported.
        _, ar_ref_X_mean, ar_ref_X_std = utils.normalize_cont(ar_ref_X)
        if same_label:
            ar_ref_X = ar_ref_X[ar_ref_y==self.original_pred]
        ar_ref_X = np.vstack([ar_ref_X, self.original_record])
        # normalize
        ar_ref_X_norm, _, _ = utils.normalize_cont(ar_ref_X, ar_mean=ar_ref_X_mean, ar_std=ar_ref_X_std)
        # logging.debug('')
        return ar_ref_X_norm, ar_ref_X_mean, ar_ref_X_std
    
    def get_positive_idx(self):
        positive_idx = self.ar_searched_index[self.ar_label[self.ar_searched_index]==1]
        return positive_idx
        
    def get_negative_idx(self):
        negative_idx = self.ar_searched_index[self.ar_label[self.ar_searched_index]==0]
        return negative_idx
        

def calc_ice(pdc_params, record, ar_ref_X=None, ar_p_min=None, ar_p_max=None):
    '''
    w/o active learning
    Args:
        pdc_params (PdcParams):
        record (2d-array):
        ar_ref_X (2d-array): default=None
        ar_p_min (1d-array): default=None
        ar_p_max (1d-array): default=None
    Return:
        pdc_params (PdcParams): 
    '''
    # check
    if ((ar_ref_X is None)&(ar_p_min is None))|((ar_ref_X is None)&(ar_p_max is None)):
        raise Exception('Either reference data or min/max value should be defined')

    pdc_params.set_record(record)
    pdc_params.set_range(ar_ref_X=ar_ref_X, ar_p_min=ar_p_min, ar_p_max=ar_p_max)
    pdc_params.calc_perturbed_values()

    # dataset construction for pred
    ar_X_ice = pdc_params.make_X_ice_data()
    
    # pred
    pdc_params.set_pred(ar_X_ice)
    # logging.debug('')
    return pdc_params

def calc_p_mice(pdc_params, record, ar_ref_X, same_label=True, ar_ref_y=None, ar_p_min=None, ar_p_max=None, 
               n_neighbors=4, mean_weight='exp_euclidean1'):
    '''
    Args:
        pdc_params (PdcParams):
        record (2d-array):
        ar_ref_X (2d-array): default=None
        same_label (bool):
        ar_ref_y (1d-array[int]): 
        ar_p_min (1d-array): default=None
        ar_p_max (1d-array): default=None
        n_neighbors (int): default=4
        mean_weight (str): default='exp_euclidean1', 
                           choices=['simple', 'euclidean1', 'euclidean2', 'exp_euclidean1', 'exp_euclidean2']
    '''
    # check
    if same_label:
        if ar_ref_y is None:
            raise Exception('ar_ref_y must be set when same_label is True')
        if ar_ref_X.shape[0]!=ar_ref_y.shape[0]:
            raise Exception('Length are inconsist with ar_ref_X and ar_ref_y')
            
    # preprocessing
    pdc_params.set_record(record)
    pdc_params.set_range(ar_ref_X=ar_ref_X, ar_p_min=ar_p_min, ar_p_max=ar_p_max)
    pdc_params.calc_perturbed_values()
    
    # dataset construction for pred
    # 1. conventional ICE step
    ar_X_ice = pdc_params.make_X_ice_data()
    
    # 2. projection step
    # make reference data for kNN 
    ar_ref_X_norm, ar_ref_X_mean, ar_ref_X_std = pdc_params.make_ref_data_for_knn(ar_ref_X, same_label, ar_ref_y)
    # kNN model construction
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, ar_ref_X_norm.shape[0]), radius=1, p=2)
    nn.fit(ar_ref_X_norm)
    # search neighbor data for ICE data
    ar_X_ice_norm, _, _ = utils.normalize_cont(ar_X_ice, ar_mean=ar_ref_X_mean, ar_std=ar_ref_X_std)
    knear_dist, knear_index = nn.kneighbors(ar_X_ice_norm, return_distance=True)
    ar_X_nears_norm = np.array([ar_ref_X_norm[knear_i] for knear_i in knear_index])
    ar_X_p_mice_norm = get_ar_X_p_mice(ar_X_nears_norm, knear_dist, mean_weight)
    # recover intervention_variable values
    ar_X_p_mice_norm[:, pdc_params.iv_index] = ar_X_ice_norm[:, pdc_params.iv_index]
    # denormalize
    ar_X_p_mice = utils.denormalize_cont(ar_X_p_mice_norm, ar_mean=ar_ref_X_mean, ar_std=ar_ref_X_std)
    
    # pred
    pdc_params.set_pred(ar_X_p_mice)
    # logging.debug('')
    return pdc_params    

def calc_ice_1d_output_only(pdc_params, iv, iv_values):
    '''
    Args:
        pdc_params (PdcParams):
        iv (str): intervention variables
        iv_values (1d-array): 
    Return:
        ar_pred_proba (1d-array): 
    '''
    iv_idx = pdc_params.iv_index[pdc_params.intervention_variables.index(iv)] # array index
    ar_X_ice = deepcopy(np.repeat(pdc_params.original_record, iv_values.shape[0], axis=0))
    ar_X_ice[:, iv_idx] = iv_values  
    # pred
    ar_pred_proba = pdc_params.model.predict_proba(ar_X_ice)[:, 1]
    # logging.debug('')
    return ar_pred_proba

def get_ar_X_p_mice(ar_X_nears, knear_dist, mean_weight='simple', e=1e-50):
    '''
    Args:
        ar_X_nears (ar): 3d-array
        knear_dist (ar): 2d-array
        mean_weight (str): default='simple', choices=['simple', 'euclidean1', 'euclidean2', 'exp_euclidean1', 'exp_euclidean2']
    Return:
        ar_X_nears_mean (ar)
    '''
    if mean_weight=='simple':
        ar_X_nears_mean = ar_X_nears.mean(axis=1)
    elif mean_weight=='euclidean1':
        ar_X_nears_mean = np.array([np.average(ar_X_nears[ci], axis=0, weights=1/(knear_dist[ci]+e)) for ci in range(ar_X_nears.shape[0])])
    elif mean_weight=='euclidean2':
        ar_X_nears_mean = np.array([np.average(ar_X_nears[ci], axis=0, weights=(1/(knear_dist[ci]+e))**2) for ci in range(ar_X_nears.shape[0])])
    elif mean_weight=='exp_euclidean1':
        ar_X_nears_mean = np.array([np.average(ar_X_nears[ci], axis=0, weights=np.exp(-knear_dist[ci])) for ci in range(ar_X_nears.shape[0])])
    elif mean_weight=='exp_euclidean2':
        ar_X_nears_mean = np.array([np.average(ar_X_nears[ci], axis=0, weights=np.exp(-((knear_dist[ci])**2))) for ci in range(ar_X_nears.shape[0])])
    else:
        print('[ERROR] Unknown mean_weight')
    return ar_X_nears_mean   

def plot_phase_diagrams(pdc_params, ice1d=True, feature_names=None, fontsize=20, xlim=None, ylim=None):
    '''
    Args:
        pdc_params (PdcParams):
        ice1d (bool): default=True
        feature_names (list[str]): default=None, if set 指定したfeature_names.
        fontsize (int):
        xlim (list[float]): default=None, [xmin, xmax]
        ylim (list[float]): default=None, [ymin, ymax]
    '''
    # check
    if len(pdc_params.intervention_variables)!=2:
        raise Exception('Unsupported number of intervention variables')
    
    figure = plt.figure(figsize=(6, 6))
    gs_master = GridSpec(nrows=2, ncols=2, width_ratios=[1, 5], height_ratios=[5, 1])
    
    # phase_diagram
    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 1]) # row, col
    axes_1 = figure.add_subplot(gs_1[:, :])
    axes_1 = ax_plot_pdc_2d(axes_1, pdc_params, feature_names=feature_names,
                            fontsize=fontsize, xlim=xlim, ylim=ylim)    
    if ice1d:
        # y-axis
        gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
        axes_2 = figure.add_subplot(gs_2[:, :])   
        axes_2 = ax_plot_ice_y(axes_2, pdc_params,
                               pdc_params.intervention_variables[1], pdc_params.list_iv_values[1])
        
        # x-axis
        gs_3 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1, 1])
        axes_3 = figure.add_subplot(gs_3[:, :])
        axes_3 = ax_plot_ice_x(axes_3, pdc_params,
                               pdc_params.intervention_variables[0], pdc_params.list_iv_values[0])
        
        axes_1.axes.xaxis.set_visible(False)
        axes_1.axes.yaxis.set_visible(False) 
        # ticks, ticklabels, label
        xticks = axes_1.get_xticks()
        xticklabels = axes_1.get_xticklabels()
        xlabel = axes_1.get_xlabel()
        yticks = axes_1.get_yticks()
        yticklabels = axes_1.get_yticklabels()
        ylabel = axes_1.get_ylabel()
        # min, max
        xmin, xmax = axes_1.get_xlim()
        ymin, ymax = axes_1.get_ylim() 
        
        # edit axes_2    
        axes_2.set_yticks(yticks)
        axes_2.tick_params(axis='y', labelsize=fontsize*0.75)
        axes_2.tick_params(axis='x', labelsize=fontsize*0.75)
    #     axes_2.set_yticklabels(yticklabels)
        axes_2.set_ylabel(ylabel, fontsize=fontsize)
        axes_2.set_ylim(ymin, ymax)

        # edit axes_3
        axes_3.set_xticks(xticks)
        axes_3.tick_params(axis='x', labelsize=fontsize*0.75)
        axes_3.tick_params(axis='y', labelsize=fontsize*0.75)
    #     axes_3.set_xticklabels(xticklabels)
        axes_3.set_xlabel(xlabel, fontsize=fontsize)
        axes_3.set_xlim(xmin, xmax)
        
        plt.subplots_adjust(hspace=0.07, wspace=0.07)
        
    plt.show()

def ax_plot_pdc_2d(ax, pdc_params, feature_names=None, fontsize=20, xlim=None, ylim=None):
    '''
    Args:
        ax (Axis):
        pdc_params (PdcParams):
        feature_names (list[str]): default=None
        font_size (int): default=16
        xlim (list[float]): default=None
        ylim (list[float]): default=None
    Return:
        ax (Axis):
    '''
    # check
    if feature_names is None:
        feature_names = pdc_params.intervention_variables

    # preprocessing
    ar_step = pdc_params.ar_step
    positive_idx = pdc_params.get_positive_idx()
    negative_idx = pdc_params.get_negative_idx()
    ar_iv_perturbed_values = pdc_params.ar_iv_perturbed_values
    ar_original_iv_values = pdc_params.ar_original_iv_values
    
    pos_left_points = ar_iv_perturbed_values[positive_idx][:,0] - (ar_step[0] / 2)
    pos_lower_points = ar_iv_perturbed_values[positive_idx][:,1] - (ar_step[1] / 2)
    neg_left_points = ar_iv_perturbed_values[negative_idx][:,0] - (ar_step[0] / 2)
    neg_lower_points = ar_iv_perturbed_values[negative_idx][:,1] - (ar_step[1] / 2)     
    
    # positive
    for pos_point in zip(pos_left_points, pos_lower_points):
        ax.add_patch(patches.Rectangle(xy=pos_point, width=ar_step[0], height=ar_step[1],
                                       fc='red', alpha=0.3, zorder=1))   
    # negative
    for neg_point in zip(neg_left_points, neg_lower_points):
        ax.add_patch(patches.Rectangle(xy=neg_point, width=ar_step[0], height=ar_step[1],
                                       fc='blue', alpha=0.3, zorder=1))   
    # original
    ax.scatter(ar_original_iv_values[0], ar_original_iv_values[1], c='green', marker='*', s=300, zorder=2)
    
    # ticks and labels
    ax.tick_params(axis='x', labelsize=fontsize*0.75)
    ax.tick_params(axis='y', labelsize=fontsize*0.75)
    ax.set_xlabel(feature_names[0], fontsize=fontsize)
    ax.set_ylabel(feature_names[1], fontsize=fontsize)
    
    # plot area
    if (xlim is None):
        ax.set_xlim(min(pdc_params.list_iv_values[0])-ar_step[0],
                    max(pdc_params.list_iv_values[0])+ar_step[0])
    else:
        ax.set_xlim((xlim[0]-ar_step[0], xlim[1]+ar_step[0]))
    if (ylim is None):
        ax.set_ylim(min(pdc_params.list_iv_values[1])-ar_step[1],
                    max(pdc_params.list_iv_values[1])+ar_step[1]) 
    else:
        ax.set_ylim((ylim[0]-ar_step[1], ylim[1]+ar_step[1]))   

    # axis
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(3)
    return ax    
    
def ax_plot_ice_y(ax, pdc_params, iv, iv_values):
    '''
    Args:
        ax (Axis):
        pdc_params (PdcParams):
        iv_values (1d-array):
    Return:
        ax (Axis):
    '''
    # ice
    ar_pred_proba = calc_ice_1d_output_only(pdc_params, iv, iv_values)
    
    ax.plot(ar_pred_proba, iv_values, linewidth=2)
    ax.axvline(pdc_params.threshold, color='grey', linestyle=':', linewidth=1.5)    
    
    # axis    
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(3) 
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])        
    return ax

def ax_plot_ice_x(ax, pdc_params, iv, iv_values):
    '''
    Args:
        ax (Axis):
        pdc_params (PdcParams):
        iv_values (1d-array):
    Return:
        ax (Axis):
    '''    
    # ice
    ar_pred_proba = calc_ice_1d_output_only(pdc_params, iv, iv_values)
    
    ax.plot(iv_values, ar_pred_proba, linewidth=2)
    ax.axhline(pdc_params.threshold, color='grey', linestyle=':', linewidth=1.5)    
    
    # axis    
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(3) 
    
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0.5', ''])        
    return ax
