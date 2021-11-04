# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:49:30 2021

@author: paimhof
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


path = 'P:/pf/pfstud/II_Group3/'

train_dset = 'dataset_rgb_nir_train.hdf5'
test_dset = 'dataset_rgb_nir_test.hdf5'



h5 = h5py.File(path+train_dset, 'r')

print(h5.keys())

#%%
img1h5 = h5["INPT_1"]
img2h5 = h5["INPT_2"]
img3h5 = h5["INPT_3"]
img4h5 = h5["INPT_4"]
GTh5 = h5["GT"]

img = np.zeros((4,10980,10980,4))
img[0,:,:,:] = img1h5[:,:,:]
img[1,:,:,:] = img2h5[:,:,:]
img[2,:,:,:] = img3h5[:,:,:]
img[3,:,:,:] = img4h5[:,:,:]



#%% create new h5 file

f = h5py.File(path+"test_file.hdf5","w");


f.create_dataset("INPT", data = img)
f.create_dataset("GT", data = GTh5[:,:])


#%% check if the new file has the correct shape 

INPT = h5py.File('P:/pf/pfstud/II_Group3/test_file.hdf5','r')

img = INPT["INPT"]
gt = INPT["GT"]
#%%
INPT.close()