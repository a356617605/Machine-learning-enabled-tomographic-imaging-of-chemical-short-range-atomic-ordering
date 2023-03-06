# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:28:12 2019
2020/02/28 plot 3D 
2020/03/17 coding algorithm to computer the probability of each voxel, 2D 
2020/03/23 coding algorithm to computer the probability of each voxel, 3D 
2020/03/27 128 neuro, 0.98 probability  
2020/04/01 recording running time
2020/04/03 send to Leigh for optimizing the code
2020/05/25 To FeAl
2020/05/25 conv_shape_a = int((img_dim_a-voxel)/stride)+1  # 2 is modified into stride
2020/11/06 extend to ZSDM for FeAl
2021/01/07 revising input formate 
2021/01/28 to FeAl 51698 1day
@author: yue.li
"""

import numpy as np
# import matplotlib.pyplot as plt
from scipy import spatial
import pandas as pd
# import random as rd
# import math
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import MaxNLocator
# import matplotlib as mpl
import datetime
from os import listdir
from itertools import product
from joblib import Parallel, delayed
#%% Function
def SRO_extract(i_threshold):
    lower, upper = threshold+i_threshold*ite, threshold+(i_threshold+1)*ite
    print('Threshold is from ', lower, ' to ', upper)
    dspp_filter = dspp.loc[(dspp['d'] > lower) & (dspp['d'] <=upper) , ['a','b','c','d']]  
    tree = []
    tree = spatial.cKDTree(df_1[['a','b','c']])
    index_voxel_sphere = tree.query_ball_point(dspp_filter[['a','b','c']], stride/2*1.414, n_jobs = -1)     
    myList = range(0,len(index_voxel_sphere))
    pre_extract_v1 = pd.DataFrame(np.arange(4).reshape(1,4),columns=['a','b','c','d'])
    for i in myList:
        df_2 = df_1.iloc[index_voxel_sphere[i,]]
        pre_extract_v1 = pre_extract_v1.append(df_2)
    pre_extract_v1 = pre_extract_v1[1:].drop_duplicates().values
    np.savetxt('pre_extract\\'+ element_name_chosen+element_name_chosen +
               '_10_10_%s_%snm_above_%s_%s_win_%s_stride_%s.csv'
               %(int(data_min['c']), int(data_max['c']), lower, upper, voxel, stride) 
               , pre_extract_v1, delimiter=',', fmt='%.3f')
    return pre_extract_v1


def SRO_extract_only_lower(lower):
    print('Threshold is above ', lower)
    dspp_filter = dspp.loc[(dspp['d'] > lower) , ['a','b','c','d']]  
    tree = []
    tree = spatial.cKDTree(df_1[['a','b','c']])
    index_voxel_sphere = tree.query_ball_point(dspp_filter[['a','b','c']], stride/2*1.414, n_jobs = -1)     
    myList = range(0,len(index_voxel_sphere))
    pre_extract_v1 = pd.DataFrame(np.arange(4).reshape(1,4),columns=['a','b','c','d'])
    for i in myList:
        df_2 = df_1.iloc[index_voxel_sphere[i,]]
        pre_extract_v1 = pre_extract_v1.append(df_2)
    pre_extract_v1 = pre_extract_v1[1:].drop_duplicates().values
    np.savetxt('pre_extract\\'+ element_name_chosen+element_name_chosen +
               '_10_10_%s_%snm_above_%s_win_%s_stride_%s.csv'
               %(int(data_min['c']), int(data_max['c']), lower, voxel, stride) 
               , pre_extract_v1, delimiter=',', fmt='%.3f')
    return pre_extract_v1
#%% Input APT data
folder = 'data'

# for filename in tqdm(os.listdir(folder)):    
#     print(filename)
#     pos = readpos(folder+'/'+filename)
#     dpos = pd.DataFrame({'x':pos['x'],
#                                 'y': pos['y'],
#                                 'z': pos['z'],
#                                 'Da': pos['m']})
#     dpos.to_csv(folder+'/'+'{}.csv'.format(filename), index=False)  

data_name = [file for file in listdir(folder) if file.endswith('.csv')]
data_name = data_name[0]
df_1 = pd.read_csv(folder+'/'+data_name, names=['a','b','c','d'])
# df_1.columns =
#%% get max, min of each column
data_min = df_1.min()
data_min['c'], data_min['b'], data_min['a'] = -180, -10, -10  #nm
data_max = df_1.max()
data_max['c'], data_max['b'], data_max['a'] = 0, 10, 10   #nm

#%% define voxel and stride 
voxel = 1   #nm   delta
stride = 0.5   #nm   delta_voxels
count = 0*8
threshold = 2.75
ite = 0.25

element_1_name = 'Co'
element_2_name = 'Cr'
element_3_name = 'Ni'
# dict = {'CoCo': 1} #dict = {'B2': 1, 'D03': 2}

for element_name_chosen in (element_1_name, element_2_name, element_3_name):
    #Inout the probability of each voxel
    data_predictions = np.load('CoCrNi_test_data_'+element_name_chosen+element_name_chosen+'_%s_%s_%s_%s_%s_%s.npy'%(int(data_min['c']), int(data_max['c']), int(data_max['b']), int(data_max['a']), voxel, stride))
    # row_number = df_1.shape[0]
    # detect_eff = 0.52
    n_dim = data_predictions.shape[1] #0 BCC 1 B2 2 D03
    #%% reshape data_predictions
    img_dim_a = int((data_max['a']-data_min['a'])/1)
    img_dim_b = int((data_max['b']-data_min['b'])/1)
    img_dim_c = int((data_max['c']-data_min['c'])/1)   
    conv_shape_a = int((img_dim_a-voxel)/stride)+1
    conv_shape_b = int((img_dim_b-voxel)/stride)+1
    conv_shape_c = int((img_dim_c-voxel)/stride)+1
    
    conv_probability = np.zeros([conv_shape_c, conv_shape_b, conv_shape_a])   
    
    # for sro_type in dict:   
    #     print(dict[sro_type])
    conv_probability[:,:,:] = np.reshape(data_predictions[:, 1], (conv_shape_c, conv_shape_b, conv_shape_a))
    # print(conv_probability[:,:,:,len1])
    #%% calculate the probability of each voxel (stride*stride*4) 
    starttime = datetime.datetime.now()
    
    pixel_probability = np.zeros([int(img_dim_c/stride), int(img_dim_b/stride), int(img_dim_a/stride)])    #1
    pixel_z = 0
    pixel_y = 0
    pixel_x = 0
    count = 0
    intervial= int(voxel/stride)
    # print ('count', count)
    # print('pixel_x=', pixel_x)
    # print('pixel_y=', pixel_y)
    # print('pixel_z=', pixel_z)
    # for pixel_dim in range (n_dim):
    #     print ('Dimension=', pixel_dim)
    for data_Z in np.arange(int(data_min['c']), int(data_max['c']), stride):
        if data_Z+stride > data_max['c']:
            continue
        else:
            for data_Y in np.arange(int(data_min['b']),int(data_max['b']), stride):
                if data_Y+stride > data_max['b']:
                    continue
                else:
                    for data_X in np.arange(int(data_min['a']), int(data_max['a']), stride):
                        if data_X+stride > data_max['a']:
                            continue
                        else:
    
                            for k in range (pixel_z, pixel_z-intervial, -1):
                                if k < 0 or k >= conv_shape_c: 
                                    continue
                                else:
                                    for j in range (pixel_y, pixel_y-intervial, -1):
                                        if j < 0 or j >= conv_shape_b:
                                            continue
                                        else:                                       
                                            for i in range (pixel_x, pixel_x-intervial, -1):
                                                if i < 0 or i >= conv_shape_a:
                                                    continue
                                                else:
                                                
                                                    temp = conv_probability[k, j, i]
                                                    pixel_probability[pixel_z, pixel_y, pixel_x] = pixel_probability[pixel_z, pixel_y, pixel_x] + temp
                                                    # print ('i=',i, "j=", j, "k=", k)
                                                    count = count + 1
                            pixel_probability[pixel_z, pixel_y, pixel_x] = pixel_probability[pixel_z, pixel_y, pixel_x]
                            # print ('count', count)
                            count=0        
                            # print ("------>>")
                            if pixel_x == int(img_dim_a/stride-1):
                                pixel_x = 0  
                            else:
                                pixel_x = pixel_x + 1
                            # print('pixel_x=', pixel_x)
                            # print('pixel_y=', pixel_y)
                            # print('pixel_z=', pixel_z)
                    if pixel_y == int(img_dim_b/stride-1):
                        pixel_y = 0                        
                    else:
                        pixel_y = pixel_y + 1
                    # print('pixel_x=', pixel_x)
                    # print('pixel_y=', pixel_y)
                    # print('pixel_z=', pixel_z)
            if pixel_z == int(img_dim_c/stride-1):
                pixel_z = 0                        
            else:
                pixel_z = pixel_z + 1
            # print('pixel_x=', pixel_x)
            # print('pixel_y=', pixel_y)
            # print('pixel_z=', pixel_z)
    endtime = datetime.datetime.now()
    print ('1st running time is', endtime-starttime)
    #%% Corresponding the probability of each small voxel with each data point, and then saved into .txt file
    starttime = datetime.datetime.now()
    
    dim_a = int((data_max['a']-data_min['a'])/stride)
    dim_b = int((data_max['b']-data_min['b'])/stride)
    dim_c = int((data_max['c']-data_min['c'])/stride)
    pixel_probability_flatten=np.reshape(pixel_probability, (dim_a*dim_b*dim_c, 1))
    np.save('pixel_probability_flatten_'+element_name_chosen+element_name_chosen, pixel_probability_flatten)
    
    try:
        data_sphere_points = np.load('data_sphere_points_cocrni_%s_%s.npy'%(int(data_min['c']), int(data_max['c'])))
    except:
        data_Z_list = list(np.arange(int(data_min['c']), int(data_max['c']), stride))
        data_Y_list = list(np.arange(int(data_min['b']), int(data_max['b']), stride))
        data_X_list = list(np.arange(int(data_min['a']), int(data_max['a']), stride))
        
        data_sphere_points = np.zeros((1,3))
        for data_Z, data_Y, data_X in product(data_Z_list, data_Y_list, data_X_list):
            if data_Z+stride > data_max['c'] or data_Y+stride > data_max['b'] or data_X+stride > data_max['a']:
                continue
            else:
                temp = np.array([data_X+stride/2, data_Y+stride/2, data_Z+stride/2]).reshape((1,3)) 
                data_sphere_points = np.concatenate((data_sphere_points, temp), axis=0)
        data_sphere_points = data_sphere_points[1:]
        np.save ('data_sphere_points_cocrni_%s_%s.npy'%(int(data_min['c']), int(data_max['c'])), data_sphere_points)
        
    data_sphere_points_probability = np.concatenate((data_sphere_points, pixel_probability_flatten), axis=1)
    dspp = pd.DataFrame(data_sphere_points_probability, columns=['a','b','c','d'])
    
    #output using different thresholds
    # myList = range(0,6)
    # pre_extract = Parallel(n_jobs=-1, verbose=1)(delayed(SRO_extract)(i) for i in myList)
    
    SRO_extract_only_lower(3.75)  

    endtime = datetime.datetime.now()
    print ('2nd running time is', endtime-starttime)

