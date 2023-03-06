# # -*- coding: utf-8 -*-
# """
# 2021/11/19 Parallel run
# @author: yue.li
# """

import os
from functools import partial
# import datetime
import pandas as pd
import numpy as np
from scipy import spatial
import multiprocessing as mp
import shutil
import tensorflow as tf
import psutil
n_phys_cores=psutil.cpu_count(logical=False)

# custom modules
import zsdm_utils_v2 as zsdm_utils
# import datasphere.populate as dsp

#%% generate exp ZSDM
folder = os.path.join(os.getcwd(), 'random_data')
csv_paths = zsdm_utils.get_csv_paths(folder)
data = pd.read_csv(csv_paths[0],
                    names=['x', 'y', 'z', 'Da'])
#print(data)
rrange_file = 'R5096_70905.RRNG'
ions, rrngs = zsdm_utils.read_rrng(rrange_file)
#print(rrngs)

element_1_name = 'Co'
element_2_name = 'Cr'
element_3_name = 'Ni'
element_1_range = rrngs[rrngs['comp']==element_1_name+':1']
element_2_range = rrngs[rrngs['comp']==element_2_name+':1']
element_3_range = rrngs[rrngs['comp']==element_3_name+':1']
#print(element_1_range)
#print(element_2_range)

element_1 = zsdm_utils.atom_filter(data, element_1_range)
element_2 = zsdm_utils.atom_filter(data, element_2_range)
element_3 = zsdm_utils.atom_filter(data, element_3_range)

data_min = data.min()
data_max = data.max()

#print(data_min)
#print(data_max)
data_min['c'], data_min['b'], data_min['a'] = -180, -10, -10  #nm
data_max['c'], data_max['b'], data_max['a'] = 0, 10, 10   #nm

# z_limit, y_limit, x_limit,  = -90, -10, -10
# scanning parameters 
voxel = np.array([1],dtype=np.float64)
#print("voxel shape {}".format(voxel.shape))
stride = 0.5
#%% Model predictions
for element_name_chosen in (element_1_name, element_2_name, element_3_name):
    nzSDM_exp_preprocessing_parallel_2 = np.load('random_nzSDM_exp_preprocessing_2_'+element_name_chosen+element_name_chosen+'_%s_%s_10_10_1_0.5.npy'
                                                 %(int(data_min['c']), int(data_max['c'])))
    w1=nzSDM_exp_preprocessing_parallel_2.shape[0]
    len1=93
    x_exp = np.reshape(nzSDM_exp_preprocessing_parallel_2, (w1, len1))
    x_exp_dim = np.reshape(x_exp[:,:], x_exp[:,:].shape+(1,))
    
    y_exp_prediction_0, y_exp_prediction_1 = np.zeros((w1, 1)), np.zeros((w1, 1)) #revised into four classes
    
    for k in range(5):
        #calling model 
        print('k=', k)
        model = tf.keras.models.load_model('saved_models_1CNN_layer_0923/model_%s.h5'%(k)) 
        model.summary()
         
        # # Test data
        y_exp_prediction_single = model.predict(x_exp_dim[:,:,:])
        y_exp_prediction_0 = np.concatenate((y_exp_prediction_0, y_exp_prediction_single[:, 0].reshape(-1,1)), axis=1)
        y_exp_prediction_1 = np.concatenate((y_exp_prediction_1, y_exp_prediction_single[:, 1].reshape(-1,1)), axis=1)
        # y_exp_prediction_2 = np.concatenate((y_exp_prediction_2, y_exp_prediction_single[:, 2].reshape(-1,1)), axis=1)
        # y_exp_prediction_3 = np.concatenate((y_exp_prediction_3, y_exp_prediction_single[:, 3].reshape(-1,1)), axis=1)
        
    y_exp_prediction_0, y_exp_prediction_1 = y_exp_prediction_0[:, 1:], y_exp_prediction_1[:, 1:]
    y_exp_prediction_0_ave, y_exp_prediction_0_std = np.mean(y_exp_prediction_0, axis=1), np.std(y_exp_prediction_0, axis=1)
    y_exp_prediction_1_ave, y_exp_prediction_1_std = np.mean(y_exp_prediction_1, axis=1), np.std(y_exp_prediction_1, axis=1)
    # y_exp_prediction_2_ave, y_exp_prediction_2_std = np.mean(y_exp_prediction_2, axis=1), np.std(y_exp_prediction_2, axis=1)
    # y_exp_prediction_3_ave, y_exp_prediction_3_std = np.mean(y_exp_prediction_3, axis=1), np.std(y_exp_prediction_3, axis=1)
    
    y_exp_predictions = np.concatenate((y_exp_prediction_0_ave.reshape(-1,1), y_exp_prediction_1_ave.reshape(-1,1)), axis=1)
    
    np.save('random_CoCrNi_test_data_'+element_name_chosen+element_name_chosen+'_%s_%s_%s_%s_%s_%s.npy'%(int(data_min['c']), int(data_max['c']), 
                                                                            int(data_max['b']), int(data_max['a']), 1, stride), y_exp_predictions)
    
