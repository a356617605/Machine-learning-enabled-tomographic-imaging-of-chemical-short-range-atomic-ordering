# -*- coding: utf-8 -*-
"""
2021/11/19 Parallel run
2021/12/07 ZSDM_Different elements
@author: yue.li
"""

import os
from functools import partial
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import multiprocessing as mp
import shutil
from itertools import product
import psutil
n_phys_cores=psutil.cpu_count(logical=False)

# custom modules
import zsdm_utils_v2 as zsdm_utils
import datasphere.populate as dsp
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
voxel = np.array([1.0],dtype=np.float64)
#print("voxel shape {}".format(voxel.shape))
stride = 0.5

#%%data sphere points
try:
    data_sphere_points = np.load("data_sphere_points_cocrni.npy")
except:
    data_Z_list = np.arange(int(data_min['c']), int(data_max['c']), stride,dtype=np.float64)
    #print("data Z list shape {}".format(data_Z_list.shape))
    data_Y_list = np.arange(int(data_min['b']), int(data_max['b']), stride,dtype=np.float64)
    data_X_list = np.arange(int(data_min['a']), int(data_max['a']), stride,dtype=np.float64)
    #print("LEN Z Y X lists {} {} {}".format(len(data_Z_list), len(data_Y_list), len(data_X_list)))
    ZZ=zsdm_utils.low_pass_filter(data_Z_list, data_max['c']-voxel) # can be improved
    #print("ZZ list shape {}".format(ZZ.shape))
    YY=zsdm_utils.low_pass_filter(data_Y_list, data_max['b']-voxel)
    XX=zsdm_utils.low_pass_filter(data_X_list, data_max['a']-voxel)
    print("LEN Z Y X lists {} {} {}".format(len(ZZ), len(YY), len(XX)))
    
    Nrows = ZZ.size * YY.size * XX.size
    data_sphere_points=dsp.init_data_sphere(XX,YY,ZZ,voxel)
    data_sphere_points[:, [0, 2]] = data_sphere_points[:, [2, 0]]
    #print("data sphere points")
    # print(data_sphere_points[0:80,])
    # print(data_sphere_points[-2,])
    np.save ("data_sphere_points_cocrni.npy", data_sphere_points)
    # data_sphere_points = np.load("data_sphere_points_cocrni.npy")
#%%Building sphere heart array
# data_Z_list = list(np.arange(int(data_min['c']), int(data_max['c']), stride))
# data_Y_list = list(np.arange(int(data_min['b']), int(data_max['b']), stride))
# data_X_list = list(np.arange(int(data_min['a']), int(data_max['a']), stride))
# data_sphere_points = np.zeros((1,3))
# for data_Z, data_Y, data_X in product(data_Z_list, data_Y_list, data_X_list):
#     if data_Z+voxel > data_max['c'] or data_Y+voxel > data_max['b'] or data_X+voxel > data_max['a']:
#         continue
#     else:
#         temp = np.array([data_X+voxel/2, data_Y+voxel/2, data_Z+voxel/2]).reshape((1,3)) 
#         data_sphere_points = np.concatenate((data_sphere_points, temp), axis=0)
# data_sphere_points = data_sphere_points[1:]
# np.save ("data_sphere_points_cocrni.npy", data_sphere_points)
# data_sphere_points = np.load("data_sphere_points_cocrni.npy")
#%%    
npy_filenames=[
      "".join(["random_zSDM_exp_test_"+element_1_name+element_1_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"]),
      "".join(["random_zSDM_exp_test_"+element_2_name+element_2_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"]),
      "".join(["random_zSDM_exp_test_"+element_3_name+element_3_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"])]

#ZSDMs
# try:
#   np.load(npy_filenames[0])
#   np.load(npy_filenames[1])
#   np.load(npy_filenames[2])
# except:
items_test = ['element_1', 'element_2', 'element_3']
from itertools import combinations
for chosen in combinations(items_test, 1):
  print(chosen[0])

items = [element_1.values, element_2.values, element_3.values]
# items = [df_1_new_element_2.values, df_1_new_element_3.values]
from itertools import combinations
num = 0
for chosen in combinations(items, 1):
  print('Cycle is', num)
  element_chosen_1, element_chosen_2= chosen[0], chosen[0]
  print(element_chosen_1, element_chosen_2)
  #element_chosen_1, element_chosen_2 = element_1.values, element_2.values
  tree_1 = cKDTree(element_chosen_1)
  tree_2 = cKDTree(element_chosen_2)
  # tree_3 = cKDTree(element_chosen_3)
  index_voxel_sphere_1 = tree_1.query_ball_point(data_sphere_points, voxel[0]*1.5/2)
  index_voxel_sphere_2 = tree_2.query_ball_point(data_sphere_points, voxel[0]*1.5/2)
  # index_voxel_sphere_3 = tree_3.query_ball_point(data_sphere_points, voxel[0]/2)
  #print(type(index_voxel_sphere))
  #print(type(index_voxel_sphere[0]))

  ZSDM_partial=partial(zsdm_utils.zsdm,
                      element_chosen_1 = element_chosen_1,
                      element_chosen_2 = element_chosen_2,
                      index_voxel_sphere_1 = index_voxel_sphere_1,
                      index_voxel_sphere_2 = index_voxel_sphere_2)

  print("starting parallel pool")
  with mp.Pool(processes=n_phys_cores) as pool:
      zSDM_output=pool.map(ZSDM_partial, range(len(data_sphere_points)))
  print("done parallel pool")

  #zSDM save
  #print(npy_filenames)
  #print(len(npy_filenames))
  zsdm_utils.ndarray2npy(zSDM_output, npy_filenames[num])
  num = num+1
#%% Data preprocessing
npy_filenames_preprocess=[
    "".join(["random_nzSDM_exp_preprocessing_2_"+element_1_name+element_1_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"]),
    "".join(["random_nzSDM_exp_preprocessing_2_"+element_2_name+element_2_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"]),
    "".join(["random_nzSDM_exp_preprocessing_2_"+element_3_name+element_3_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"])]

npy_filenames_prediction=[
    "".join(["random_CoCrNi_test_data_"+element_1_name+element_1_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"]),
    "".join(["random_CoCrNi_test_data_"+element_2_name+element_2_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"]),
    "".join(["random_CoCrNi_test_data_"+element_3_name+element_3_name+"_",str(int(data_min['c'])),"_",str(int(data_max['c'])),"_10_10_1_0.5.npy"])]
# import tensorflow as tf

for element_chosen in range(3):
    print(element_chosen)
    # if element_chosen==0 or element_chosen==1:
    #   continue
    # else:
    zSDM_exp_element_1 = np.load(npy_filenames[element_chosen])
    # zSDM_exp_element_2 = np.load(npy_filenames[1])
    # zSDM_exp_element_3 = np.load(test_dir+"zSDM_exp_test_"+element_3_name+element_3_name+"_%s_%s_%s_%s_%s_%s.npy"
    #               %(int(data_min['c']), int(data_max['c']), int(data_max['b']), int(data_max['a']), voxel, stride) )            

    dim = zSDM_exp_element_1.shape[0]*zSDM_exp_element_1.shape[2]
    zSDM_exp_element_1_2d = np.reshape(np.transpose(zSDM_exp_element_1,(1,0,2)),(-1, dim))
    # zSDM_exp_element_2_2d = np.reshape(np.transpose(zSDM_exp_element_2,(1,0,2)),(-1, dim))
    # zSDM_exp_element_3_2d = np.reshape(np.transpose(zSDM_exp_element_3,(1,0,2)),(-1, dim))
    # Data normalization
    nzSDM_exp_element_1 = zsdm_utils.normdata(zSDM_exp_element_1_2d)
    # nzSDM_exp_element_2 = normdata(zSDM_exp_element_2_2d)
    # nzSDM_exp_element_3 = normdata(zSDM_exp_element_3_2d)

    nzSDM_exp= nzSDM_exp_element_1 #revised into three elements
    save_newexpZSDMs = False
    # Build ouptfile
    if save_newexpZSDMs == True:
        try:
            shutil.rmtree('Results_newexpZSDMs_'+element_1_name)
        except:
            print("file does not exist")
        os.mkdir('Results_newexpZSDMs_'+element_1_name)

    (len1,w1) = np.shape(nzSDM_exp)

    preprocessing_partial=partial(zsdm_utils.exp_data_processing_parallel,
                            len1=len1, 
                            nzSDM_exp=nzSDM_exp, 
                            save_newexpZSDMs=save_newexpZSDMs, 
                            zSDM_exp_element_1_2d=zSDM_exp_element_1_2d, 
                            element_1_name=element_1_name)

    print("starting parallel pool")
    with mp.Pool(processes=n_phys_cores) as pool:
        nzSDM_exp_preprocessing_parallel_2=pool.map(preprocessing_partial, range(w1))
    print("done parallel pool")
    zsdm_utils.ndarray2npy(nzSDM_exp_preprocessing_parallel_2, npy_filenames_preprocess[element_chosen])
