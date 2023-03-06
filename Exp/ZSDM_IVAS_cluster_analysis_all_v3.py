# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:30:19 2021

@author: yue.li
"""
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from os import listdir
import matplotlib.pyplot as plt
from collections import Counter
import re
from scipy import stats


def atom_filter(x, Atom_range):
    Atom_total = pd.DataFrame()
    for i in range(len(Atom_range)):
        Atom = x[x['Da'].between(Atom_range['lower'][i], Atom_range['upper'][i], inclusive=True)]
        Atom_total = Atom_total.append(Atom)
        # Count_Atom= len(Atom_total['Da'])   
    return Atom_total[['x','y','z']]

def read_rrng(f):
    rf = open(f,'r').readlines()
    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')
    ions = []
    rrngs = []
    for line in rf:
        m = patterns.search(line)
        if m:
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            else:
                rrngs.append(m.groups()[2:])
    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True) 
    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)
    return ions, rrngs

def clipped_volume(type_volume):
    if type_volume ==0:
        # Volume from all data
        folder = 'data'
        data_name = [file for file in listdir(folder) if file.endswith('.csv')]
        data_name = data_name[0]
        df_1 = pd.read_csv(folder+'/'+data_name, names=['x', 'y', 'z', 'Da'])
        data_min_orig = df_1.min()
        data_max_orig = df_1.max()
        df_1_upper = df_1[df_1['z']>-1]
        data_min_orig_upper = df_1_upper.min()
        data_max_orig_upper = df_1_upper.max()
        
        df_1_element_1 = atom_filter(df_1, element_1_range)
        df_1_element_2 = atom_filter(df_1, element_2_range)
        df_1_element_3 = atom_filter(df_1, element_3_range)
        
        #calculate radius and volume 
        data_radius_size = (abs(data_min_orig)+abs(data_max_orig))/2
        data_radius_size_upper = (abs(data_min_orig_upper)+abs(data_max_orig_upper))/2
        height, radius_2, radius_1 = data_radius_size['z']*2, data_radius_size['x'], data_radius_size_upper['x'] 
        volume_total_orig = 3.14*height*(radius_1**2+radius_2**2+radius_2*radius_1)/3   #nm3
        
        df_1_new = df_1.loc[(df_1['z']>=data_min['z']) & (df_1['z']<=data_max['z']) &
                  (df_1['y']>=data_min['y']) & (df_1['y']<=data_max['y']) &
                  (df_1['x']>=data_min['x']) & (df_1['x']<=data_max['x'])]
        df_1_new_element_1 = atom_filter(df_1_new, element_1_range)
        df_1_new_element_2 = atom_filter(df_1_new, element_2_range)
        df_1_new_element_3 = atom_filter(df_1_new, element_3_range)
        
        
        number_total_orig, number_total = len(df_1_element_1)+len(df_1_element_2)+len(df_1_element_3), len(df_1_new_element_1)+len(df_1_new_element_2)+len(df_1_new_element_3)
        volume_total = (number_total/number_total_orig)*volume_total_orig  #nm**3
    else:   
        # volume from a cube
        folder = 'Cube_data'
        data_name = [file for file in listdir(folder) if file.endswith('.csv')]
        data_name = data_name[0]
        df_1 = pd.read_csv(folder+'/'+data_name, names=['x', 'y', 'z', 'Da'])
        data_min_orig = df_1.min()
        data_max_orig = df_1.max()
        df_1_upper = df_1[df_1['z']>-1]
        data_min_orig_upper = df_1_upper.min()
        data_max_orig_upper = df_1_upper.max()
        
        df_1_element_1 = atom_filter(df_1, element_1_range)
        df_1_element_2 = atom_filter(df_1, element_2_range)
        df_1_element_3 = atom_filter(df_1, element_3_range)
        
        #calculate radius and volume 
        data_radius_size = (abs(data_min_orig)+abs(data_max_orig))/2
        data_radius_size_upper = (abs(data_min_orig_upper)+abs(data_max_orig_upper))/2
        height, radius_2, radius_1 = data_radius_size['z']*2, data_radius_size['x'], data_radius_size_upper['x'] 
        print(height, radius_2, radius_1)
        volume_total_cube = height*radius_1*radius_2*4  #nm3
        
        
        folder = 'data'
        data_name = [file for file in listdir(folder) if file.endswith('.csv')]
        data_name = data_name[0]
        df_1_all = pd.read_csv(folder+'/'+data_name, names=['x', 'y', 'z', 'Da'])
        df_1_all_new = df_1_all.loc[(df_1_all['z']>=data_min['z']) & (df_1_all['z']<=data_max['z']) &
                  (df_1_all['y']>=data_min['y']) & (df_1_all['y']<=data_max['y']) &
                  (df_1_all['x']>=data_min['x']) & (df_1_all['x']<=data_max['x'])]
        df_1_new_element_1 = atom_filter(df_1_all_new, element_1_range)
        df_1_new_element_2 = atom_filter(df_1_all_new, element_2_range)
        df_1_new_element_3 = atom_filter(df_1_all_new, element_3_range)
        
        
        number_total_cube= len(df_1_element_1)+len(df_1_element_2)+len(df_1_element_3) 
        number_total = len(df_1_new_element_1)+len(df_1_new_element_2)+len(df_1_new_element_3) 
        volume_total = number_total/number_total_cube*volume_total_cube #nm**3
    return volume_total

def plot3d(plot_3d, df_with_labels_filter):
    if plot_3d == True:
        fig1 = plt.figure()
        ax = plt.subplot(111, projection='3d')  # build a project
        # ax.scatter(data_np [:, 0], data_np [:, 1], data_np [:, 2], c=labels, cmap='turbo', marker='o', s=1)  # 绘制数据点
        limit = 5000
        ax.scatter(df_with_labels_filter [:limit, 0], df_with_labels_filter [:limit, 1], df_with_labels_filter [:limit, 2], 
                   c=df_with_labels_filter [:limit, 4], cmap='prism', marker='o', s=2)  # 绘制数据点
        
        # ax.tick_params(axis='both', which='major', labelsize=16)
        # plt.axis([-10, 10, -10, 10])
        ax.set_zlabel('Z (nm)', fontsize=16)  # axis
        ax.set_ylabel('Y (nm)', fontsize=16)
        ax.set_xlabel('X (nm)', fontsize=16)
        # ax.axis('auto') 
        plt.show()
#%%
#Parameters
# sro_type = 'CoCo'
threshold = 3.75
voxel = 1   #nm   delta
stride = 0.5 
plot_3d = False
#%%Input SRO data from CNN
folder = 'pre_extract'
data_name = [file for file in listdir(folder) if file.endswith('.csv')]
counts_csro = np.zeros([32, 2])
ite = 0
# data_name = data_name[0]
fig1 = plt.figure()
for data_name_chosen in data_name:
    print(data_name_chosen)
    df = pd.read_csv(folder+'/'+data_name_chosen, names=['x', 'y', 'z', 'Da'])
    data_min = df.min()
    data_max = df.max()
    data_min['z'], data_min['y'], data_min['x'] = -180, -10, -10  #nm
    data_max['z'], data_max['y'], data_max['x'] = 0, 10, 10   #nm
    
    #Finding element 1 Fe and element 2 Al    
    rrange_file = 'R5096_70905.RRNG'
    ions, rrngs = read_rrng(rrange_file)
    
    element_1_name = 'Co'
    element_2_name = 'Cr'
    element_3_name = 'Ni'
    element_1_range = rrngs[rrngs['comp']==element_1_name+':1']
    element_2_range = rrngs[rrngs['comp']==element_2_name+':1']
    element_3_range = rrngs[rrngs['comp']==element_3_name+':1']
    
    df_element_1 = atom_filter(df, element_1_range)
    df_element_2 = atom_filter(df, element_2_range)
    df_element_3 = atom_filter(df, element_3_range)
    df_np = df.values
    
    
    #%%Calculate CSRO data from DBSCAN
    df_np_preprocess = np.concatenate((df_np[:,0].reshape(-1, 1)-10, df_np[:,1].reshape(-1, 1)-10, df_np[:,2].reshape(-1, 1)), axis=1) #xyz is nm scale and do not standard scale.
    
    db = DBSCAN(eps=0.4, min_samples=3).fit(df_np_preprocess)
    labels = db.labels_
    
    #calculate the atom number of each SRO domain
    labels_del_1 = labels[labels!=-1] #Noisy samples are given the label -1
    counts = Counter(labels_del_1)
    atom_number_each_SRO = np.array(list(counts.values()))
    
    df_with_labels = np.concatenate((df_np, labels.reshape(-1, 1)), axis=1)
    df_with_labels_filter= df_with_labels[(df_with_labels[:,4]!=-1)]  
    # np.savetxt(sro_type + 
    #             '_DBSCAN_%s_%s_10_10_above_%s_win_%s_stride_%s.csv'
    #             %(int(data_min['z']), int(data_max['z']), threshold, voxel, stride) 
    #             , df_with_labels_filter[:, [0, 1, 2, 4]], delimiter=',', fmt='%.3f')
    
    #Plot 3d
    plot3d(plot_3d, df_with_labels_filter)
    
    #%%Plot count distribution
    volume_total = clipped_volume(0) #1 from a chosen cube.
    atom_number_each_SRO[atom_number_each_SRO>160]=160
    upper = atom_number_each_SRO.max()
    n, bins, patches = plt.hist(x=atom_number_each_SRO, bins=32, range=(0, 160), label=data_name_chosen[:11], alpha=0.7, rwidth=0.85)
    counts_csro[:, ite] = n
    ite+=1

    
#stastical analysis PCC    
onset = 0
stats_21, p21 = stats.pearsonr(counts_csro[onset:,0], counts_csro[onset:,1]) # actual vs random
print('stats_21 is', stats_21, ' with beginner of ', onset)

#stastical analysis Chi-square
# counts_csro[counts_csro==0] = 1
# cq_21,cqp_21 = stats.chisquare(counts_csro[onset:,0], f_exp = counts_csro[onset:,1]) # actual vs random
# cq_21_pearson = np.sqrt(cq_21/(cq_21+counts_csro.shape[0]))
# import pandas as pd
 
counts_csro_pd = pd.DataFrame(counts_csro, columns=['actual', 'random'])
result = counts_csro_pd[(counts_csro_pd['random']>=1)]
result = np.array(result, dtype=np.float64)
# result = counts_csro[np.all(counts_csro, axis=1)]
cq_21,cqp_21 = stats.chisquare(result[onset:,0], f_exp = result[onset:,1]) # actual vs random
reduced_cq_21 = cq_21/result.shape[0]
n_total= (counts_csro[:,0].sum()+counts_csro[:,1].sum())/2.0
cq_21_pearson = np.sqrt(cq_21/(cq_21+n_total))

print('cq_21 is', cq_21, ' with beginner of ', onset)
print('cq_21_pearson is', cq_21_pearson, ' with beginner of ', onset)

plt.xlabel('CSRO size (atoms)')
plt.ylabel('Counts')
plt.legend(loc='upper right')

plt.title('PCC is %.2f and cqp_21 is %2.f and cq_pearson is %.2f'%(stats_21, cqp_21, cq_21_pearson))
#%%plot number density distribution
# onset = 0
# number_density = counts_csro/volume_total*10**(27) #m^-3
# number_density = number_density.astype(float)
# np.save('pre_extract/'+data_name[0][:11]+'_'+rrange_file, number_density)
# fig1 = plt.figure()
# plt.bar(bins[onset+1:,]-5,number_density[onset:,0], width=8, label=data_name[0][:11], alpha=0.7)
# plt.bar(bins[onset+1:,]-5,number_density[onset:,1], width=8, label=data_name[1][:11], alpha=0.7)
# plt.xlabel('CSRO size (atoms)')
# plt.ylabel('Number density (m$^{-3}$)')
# plt.legend(loc='upper right')

# #stastical analysis PCC    
# stats_21, p21 = stats.pearsonr(number_density[onset:,0], number_density[onset:,1]) # actual vs random
# print('stats_21 is', stats_21, ' with beginner of ', onset)

# #stastical analysis Chi-square
# result = number_density[np.all(number_density, axis=1)]
# cq_21,cqp_21 = stats.chisquare(result[onset:,0], f_exp = result[onset:,1]) # actual vs random
# cq_21_pearson = np.sqrt(cq_21/(cq_21+result.shape[0]))

# print('cq_21 is', cq_21, ' with beginner of ', onset)
# print('cq_21_pearson is', cq_21_pearson, ' with beginner of ', onset)
# plt.title('PCC is %.2f and cq_pearson is %.2f'%(stats_21, cq_21_pearson))


#%%%
#plot
# def normdata(data):
    
#     (len1,) = np.shape(data)
#     ndata = np.zeros([len1,])
#     for i in range(len1): 
#         ndata[i,]=data[i,]/sum(data)
#         # print(j, i)
#         # print(sum(data[:,i]))
#     return ndata

# fig1 = plt.figure()
# plt.plot(bins[onset+1:,], (counts_csro[onset:,0]), label=data_name[0])
# plt.plot(bins[onset+1:,], (counts_csro[onset:,1]), label=data_name[1]) 
# plt.legend(loc='upper right')
# plt.xticks(rotation=45)
# plt.xlabel('CSRO size (atoms)')
# plt.ylabel('Counts')
    #%%Input CSRO data from IVAS
    # folder = 'Cluster_analysis_CoCo'
    # data_name = [file for file in listdir(folder) if file.endswith('.csv')]
    # data_name = data_name[0]
    # df_2 = pd.read_csv(folder+'/'+data_name, skiprows = lambda x: x in [0,1,2,3,4,5,6,7,8,9,11])
    
    # aspect_ratio = df_2["R_gy (nm) Solute'"]/df_2["R_gx (nm) Solute'"]
    # oblateness = df_2["R_gz (nm) Solute'"]/df_2["R_gy (nm) Solute'"]
    # atom_number_each_SRO = np.array(df_2["Solute Ions"], dtype=np.float64)
    
    # # Plot cluster size distribution
    # import seaborn as sns 
    # # sns.set_theme()
    # fig1 = plt.figure()
    # ax=sns.distplot(atom_number_each_SRO,color="r",bins=15,hist=True, norm_hist=False)
    # # sns.show()
    
    # # Plot morophology
    # fig1 = plt.figure(figsize=(8,6))
    # ax = fig1.add_subplot(1, 1, 1)
    # sc = plt.scatter(oblateness, aspect_ratio, s=atom_number_each_SRO*0.1, c=atom_number_each_SRO, cmap='jet',alpha=0.5)
    # cbar=plt.colorbar(sc)
    # cbar.set_label('Number of atoms', fontsize=18)
    # cbar.ax.tick_params(labelsize=18)
    # plt.xlabel('Oblateness', fontsize=18)
    # plt.ylabel('Aspect ratio', fontsize=18)
    # # font1 = {'family' : 'Arial',
    # # 'weight' : 'normal',
    # # 'size'   : 15,
    # # }
    # plt.tick_params(labelsize=18)
    # plt.xticks([0.0,0.5,1.0])
    # plt.yticks([0.0,0.5,1.0])
    # plt.vlines(0.5, 0, 1, linestyles = "dashed")
    # plt.hlines(0.5, 0, 1, linestyles = "dashed")
    # plt.text(0.8, 0.8, 'Sphere',fontsize=18)
    # plt.text(1-0.8, 0.8, 'Disc',fontsize=18)
    # plt.text(1-0.8, 1-0.8, 'Lath',fontsize=18)
    # plt.text(0.8, 1-0.8, 'Rod',fontsize=18)
    # plt.show()
    
    ##Calculate the quantitative information
    # number_density = df_2.shape[0]/volume_total*10**(27) #m^-3
    # radius_equ_py = (df_2["R_gx (nm) Solute'"]*df_2["R_gy (nm) Solute'"]*df_2["R_gz (nm) Solute'"])**(1.0/3.0)*(5.0/3)**0.5

