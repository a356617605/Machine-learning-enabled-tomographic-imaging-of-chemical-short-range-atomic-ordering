import os
import re
import glob
import numpy as np
import pandas as pd 
from fast_histogram import histogram2d as fast_histogram2d
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def low_pass_filter(arr, k):
    return arr[np.where(~(arr > k))]

def zsdm(count, 
          element_chosen_1, 
          element_chosen_2, 
          index_voxel_sphere_1, 
          index_voxel_sphere_2):
    data_voxel_sphere_1 = element_chosen_1[index_voxel_sphere_1[count,],:] # (N,3) ndarray
    data_voxel_sphere_2 = element_chosen_2[index_voxel_sphere_2[count,],:] # (N,3) ndarray
    # data_voxel_sphere_3 = element_chosen_3[index_voxel_sphere_3[count,],:] # (N,3) ndarray
    # 
    x_tot = [];
    y_tot = [];
    num_in_SDM = 0
    max_cand =0
    radius = 1.5
    SDM_bins = 200
    # no n_jobs=-1, you're already in a parallel region managed by python multiprocessing
    small_tree_1 = cKDTree(data_voxel_sphere_1)
    small_tree_2 = cKDTree(data_voxel_sphere_2)
    # small_tree_3 = cKDTree(data_voxel_sphere_3)
    
    cand = small_tree_1.query_ball_tree(small_tree_2, radius) #, tree_1 is centre
    # print("cand is", cand)
    for l in cand:
        num_in_SDM += len(l)
        if (len(l) > max_cand):
            max_cand = len(l)
    x_tot = np.zeros([num_in_SDM,], dtype = np.float64)
    y_tot = np.zeros([num_in_SDM,], dtype = np.float64)
    x = np.zeros([max_cand,], dtype = np.float64)   
    y = np.zeros([max_cand,], dtype = np.float64)
    
    start = 0
    i = 0
    for l in cand:
        length = len(l)
        x_tot[start:(start+length)] = np.ndarray.__sub__(data_voxel_sphere_2[l,0],data_voxel_sphere_1[i,0])
        y_tot[start:(start+length)] = np.ndarray.__sub__(data_voxel_sphere_2[l,2],data_voxel_sphere_1[i,2])
        i += 1
        start = start+length
    notzero = (x_tot!=0)*(y_tot!=0)
    # The previous allocation of "SDM" as zeros
    # will be deallocated here to allocate the output of fast histogram
    # and let "SDM" point to it
    # from fast_histogram import histogram2d as fast_histogram2d
    SDM = fast_histogram2d(y_tot[notzero],x_tot[notzero], range = [[-radius,radius],[-radius,radius]],  bins=SDM_bins)
    
    y_zSDM = SDM.sum(axis=1).reshape((200, 1))
    x_zSDM = np.arange(-radius, radius, radius*2/200).reshape((200, 1))
    zSDM_simu = np.concatenate((x_zSDM, y_zSDM), axis=1) # check out above how to avoid np.concatenate
    zSDM_simu_index = np.where((zSDM_simu[:,0]>=-0.7) & (zSDM_simu[:,0]<0.7))
    zSDM_simu_index_array = np.array(zSDM_simu_index).reshape(-1, 1)
    zSDM_simu_part = zSDM_simu[zSDM_simu_index_array[0, 0]:zSDM_simu_index_array[-1, 0]+1, ]
    # if count%500 == 0:
    #     fig2D = plt.figure(figsize=(4,4))                
    #     ax2D = fig2D.add_subplot(111)   
    #     plt.plot(zSDM_simu_part[: ,0], zSDM_simu_part[: ,1])
    #     # ax2D.set_aspect(1)
    #     plt.xlim((-0.7,0.7))
    #     plt.close()  
    #     fig2D.savefig('zsdm_%s.png'%(count), dpi=92)
    return zSDM_simu_part

def atom_filter(x, Atom_range):
    check=pd.Series(np.zeros(shape=(len(x),), dtype=np.bool))
    for i in range(len(Atom_range)):
        check = check | \
                x['Da'].between(Atom_range['lower'][i], Atom_range['upper'][i], inclusive=True)
    # print(check[~check].index    )
    print("ATOM TOTAL = {}".format(len(x[check][['x','y','z']])))
    return x[check][['x','y','z']]

def read_rrng(f):
    with open(f,'r') as F:
        rf = F.readlines()
    #print(rf) # list of lines
    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')
    ions = [] # will be a list of tuples
    rrngs = [] # will be a list of tuples
    for line in rf:
        m = patterns.search(line)
        if m:
            # print(m.groups()) # each group is a tuple
            if m.groups()[0] is not None: # tuples to create ions df
                ions.append(m.groups()[:2])
            else: # tuples to create rrngs df
                rrngs.append(m.groups()[2:])

    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True) 
    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)
    #print(ions.dtypes)
    #print(rrngs.dtypes)
    return ions, rrngs

def readpos(file_name):
    f = open(file_name, 'rb')
    dt_type = np.dtype({'names':['x', 'y', 'z', 'm'], 
                  'formats':['>f4', '>f4', '>f4', '>f4']})
    pos = np.fromfile(f, dt_type, -1)
    f.close()
    return pos

def get_csv_paths(path):
    return glob.glob(os.path.join(path, "*.csv"))


def ndarray2npy(arr, filename):
    print("Saving np array into {}...".format(filename))
    np.save(filename, arr)
    print("Done.")

# Data normalization from 0 to 1 for double column dataframe, returns single column array
def normdata(data):
    
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1//2])
    for i in range(w1//2):
        # if (max(data[:,2*i+1])-min(data[:,2*i+1]))==0:
        #     ndata[:,i]=data[:,2*i+1]
        # else:
        ndata[:,i]=(data[:,2*i+1]-min(data[:,2*i+1]))/(max(data[:,2*i+1])-min(data[:,2*i+1]))
        # if (max(data[:,2*i+1])-min(data[:,2*i+1]))==0: 
        #     print(ndata[:,i])
    return ndata

# Data normalization from 0 to 1 for single column dataframe
def normdatasingle(data):
    
    (len1,w1) = np.shape(data)
    ndata = np.zeros([len1,w1])
    for i in range(w1):
        ndata[:,i]=(data[:,i]-min(data[:,i]))/(max(data[:,i])-min(data[:,i]))
    return ndata

# Extracting experimental data
def exp_data_processing_parallel (i, len1, nzSDM_exp, save_newexpZSDMs, zSDM_exp_element_1_2d, element_1_name):
    n_input = 93
    dim = int(len1/n_input)
    # i=1
    # print ('dim=', dim, 'w1=', w1)
    new1_BG_remove_3 = np.zeros([len1, 1])
    for j in range(dim):
        # for i in range(w1):
        if np.isnan(np.min(nzSDM_exp[n_input*j : n_input*(j+1), i])):
            # print ("Nan")
            new1_BG_remove_3[n_input*j : n_input*(j+1), ] = nzSDM_exp[n_input*j : n_input*(j+1), i].reshape(-1,1)
            continue
        # print ('i=',i, 'j=', j)
        #savgol_filter to smooth the data
        new1 = savgol_filter(nzSDM_exp[n_input*j : n_input*(j+1), i], 15, 8)  #21
        # # Plot smoothed curve    
        # fig2D = plt.figure(figsize=(4,4))                
        # ax2D = fig2D.add_subplot(111)   
        # plt.plot(zSDM_exp_FeFe[:, 0], nzSDM_exp[n_input*j : n_input*(j+1), i], label='Original curve')
        # plt.plot(zSDM_exp_FeFe[:, 0], new1[:,], label='Smoothed curve')
        # plt.legend(loc='upper right')
        
        # Background substraction based on Moody 2009
        new1_BG_remove = np.zeros([n_input, 1])
        new1_BG_remove_2 = np.zeros([n_input, 1])
        new1_BG_remove[:,0] = new1[0:n_input, ]
        new1_BG_remove_2[0,0], new1_BG_remove_2[n_input-1,0] = new1[0,], new1[n_input-1,] 
        # new1_BG_remove_2[0,0], new1_BG_remove_2[n_input-1,0] = min(new1[0:n_input, ]), min(new1[0:n_input, ])
        
        # local minmial
        localmin = argrelextrema(new1, np.less)
        localmin_array = np.array(localmin)
        localmin_value = localmin_array[np.where(localmin_array>int(n_input/2))]
        # if new1[0,] == max(new1[0:n_input, ]):
        #     new1_BG_remove_2[0,0] = min(new1[0:n_input, ])
        # if new1[n_input-1,] == max(new1[0:n_input, ]):
        #     new1_BG_remove_2[n_input-1,0] = min(new1[0:n_input, ])  
    
        try:
            localmin_value[0,]
            ite_1=0
            while max(new1_BG_remove[int(0.5*n_input/2):int(3.0*n_input/4), 0]) > new1[localmin_value[0, ], ]:
                for m in range (1, n_input-1):
                    new1_BG_remove_2[m,] = min(new1_BG_remove[m,], (new1_BG_remove[m-1,] + new1_BG_remove[m+1,])/2)
                new1_BG_remove = new1_BG_remove_2
                # # # Plot background curve
                if ite_1 == 1000:
                    # print("Iteration is above 1000")
                    break
                # fig2D = plt.figure(figsize=(4,4))                
                # ax2D = fig2D.add_subplot(111)  
                # # plt.plot(zSDM_exp_FeFe[:, 0], nzSDM_exp[n_input*j : n_input*(j+1), i], label='Original curve')
                # plt.plot(zSDM_exp_FeFe[:, 0], new1[:,], label='Smoothed curve')
                # plt.plot(zSDM_exp_FeFe[:, 0], new1_BG_remove[:,], label='Background curve')
                ite_1=ite_1+1
                
            # print('Iteration number is ', ite_1)
            # # Plot background curve
            # fig2D = plt.figure(figsize=(4,4))                
            # ax2D = fig2D.add_subplot(111)  
            # plt.plot(zSDM_exp_FeFe[:, 0], nzSDM_exp[n_input*j : n_input*(j+1), i], label='Original curve')
            # plt.plot(zSDM_exp_FeFe[:, 0], new1[:,], label='Smoothed curve')
            # plt.plot(zSDM_exp_FeFe[:, 0], new1_BG_remove[:,], label='Background curve')
            # plt.legend(loc='upper right')
        except:
            new1_BG_remove = np.zeros((n_input, 1))
        # Data after background substraction
        new1_BG_remove_3[n_input*j : n_input*(j+1), ] = normdatasingle(new1.reshape(n_input, 1)-new1_BG_remove)  
        # Plot data after background substraction
        if save_newexpZSDMs == True:
            fig2D = plt.figure(figsize=(4,4))                
            ax2D = fig2D.add_subplot(111) 
            plt.plot(zSDM_exp_element_1_2d[:, 0], nzSDM_exp[n_input*j : n_input*(j+1), i], label='Original curve') 
            plt.plot(zSDM_exp_element_1_2d[:, 0], new1[:,], label='Smoothed curve')
            plt.plot(zSDM_exp_element_1_2d[:, 0], new1_BG_remove[:,], label='Background curve')
            plt.plot(zSDM_exp_element_1_2d[:, 0], new1_BG_remove_3[n_input*j: n_input*(j+1),:], label='Curve after BR') 
            plt.legend(loc='upper right')
            plt.close()  
            if j==0:
                fig2D.savefig('Results_newexpZSDMs_'+element_1_name+'/Newexp_ZSDM_'+element_1_name+element_1_name+'_%s.png'%(i), dpi=92)
            # if j==1:
            #     fig2D.savefig('Results_newexpZSDMs_'+element_2_name+'/Newexp_ZSDM_'+element_2_name+element_2_name+'_%s.png'%(i), dpi=92)
            # if j==2:
            #     fig2D.savefig('Results_newexpZSDMs/Newexp_ZSDM_'+element_3_name+element_3_name+'_%s.png'%(i), dpi=92)    
    return new1_BG_remove_3
