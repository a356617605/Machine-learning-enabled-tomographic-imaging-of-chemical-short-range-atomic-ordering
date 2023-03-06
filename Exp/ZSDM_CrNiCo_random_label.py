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
folder = os.path.join(os.getcwd(), 'data')
csv_paths = zsdm_utils.get_csv_paths(folder)
data = pd.read_csv(csv_paths[0],
                   names=['x', 'y', 'z', 'Da'])

#Data random
data_mass_spectrum = data['Da']
import random as rd


data_mass_spectrum_no_shuffle = data_mass_spectrum.values
row = data_mass_spectrum_no_shuffle.shape[0]
idx = rd.sample(range(row),row) 
data_mass_spectrum_shuffle = data_mass_spectrum_no_shuffle[idx]

for i in range(4):
    data_mass_spectrum_no_shuffle=data_mass_spectrum_shuffle
    row = data_mass_spectrum_no_shuffle.shape[0]
    idx = rd.sample(range(row),row) 
    data_mass_spectrum_shuffle = data_mass_spectrum_no_shuffle[idx]

data['Da_2'] = data_mass_spectrum_shuffle
data = data.drop(['Da'],axis=1)
np.savetxt ("random_data/data_random_label.csv", data, delimiter=',', fmt='%.3f')
