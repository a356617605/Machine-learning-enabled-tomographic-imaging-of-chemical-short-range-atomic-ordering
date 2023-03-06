"""
2021/08/10  CoCrNi ZSDMs Curve
@author: yue.li
"""

import numpy as np
import os
import shutil
from Euler_transformation import *
from generator_single_SDMs import *
#%% Input file and parameters
data = np.loadtxt('ggoutputFile_L12_X3Co_1nm_a_0.36.txt')
lattice_para = 0.36

sigma_xy_all = np.arange(0.2,0.8,0.05)
sigma_z_all = np.arange(0.02,0.05,0.002)
detect_eff_array = np.arange(0.2,0.7,0.05)
atomic_number_1 = 59  #Co
atomic_number_2 = 64  #others
save_XZSDM, save_ZSDM, plot_XZSDM, plot_ZSDM, plot_noise = False, True, False, False, False
Delete_folder = True  #default
image_name_1 = "CoCrNi_L12_CoCo"#Co has 2 times space than others
image_name_2 = "CoCrNi_L12_others"
#%% clean ouptfile contents
if Delete_folder == True:
    try:
        shutil.rmtree('Results_ZXSDMs') 
        shutil.rmtree('Results_ZSDMs')
    except:
        print("file does not exist")
    os.mkdir('Results_ZXSDMs') 
    os.mkdir('Results_ZSDMs')
#%% Remove all duplicates    
data_clean = np.unique(data, axis=0)
#%% Euler_transformation
data_reconstruction = Euler_transformation_100(data_clean, False)
# data_reconstruction = Euler_transformation_110(data_clean, False)
# data_reconstruction = Euler_transformation_111(data_clean, False)
#%% Add Gaussian noise
zSDM_simu_0 = np.zeros((int(0.69/0.015*2+1), 1))
zSDM_simu_1 = np.zeros((int(0.69/0.015*2+1), 1))
for sigma_xy in sigma_xy_all:
    for sigma_z in sigma_z_all:
        for detect_eff in detect_eff_array:  
            for el_number in range(2):
                zSDM_simu_part = single_SDMs(data_reconstruction, el_number, sigma_xy, sigma_z, plot_noise, detect_eff, atomic_number_1, atomic_number_2, lattice_para, image_name_1, image_name_2, save_XZSDM, plot_XZSDM, save_ZSDM, plot_ZSDM)
                if el_number==0:
                    zSDM_simu_0 = np.concatenate((zSDM_simu_0, zSDM_simu_part), axis=1)
                else:
                    zSDM_simu_1 = np.concatenate((zSDM_simu_1, zSDM_simu_part), axis=1)
zSDM_simu_0 = zSDM_simu_0[:,1:]
zSDM_simu_1 = zSDM_simu_1[:,1:]
#%%save
np.save ("zSDM_simu_"+image_name_1, zSDM_simu_0)
np.save ("zSDM_simu_"+image_name_2, zSDM_simu_1)