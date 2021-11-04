# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:15:02 2021

@author: paimhof
"""

import h5py
import numpy as np
import matplotlib.pylab as plt

path = 'P:/pf/pfshare/data/mikhailu/'
training_dset = 'dataset_rgb_nir_train.hdf5'

dset_train = h5py.File(path+training_dset,"r")

print(dset_train.keys())


INPT_1 = dset_train["INPT_1"]
NIR_1 = dset_train["NIR_1"]
# groundtruth
GT = dset_train["GT"]
Cloud = dset_train["CLD_1"]

# print(np.shape(INPT_1))
# print(np.shape(NIR_1))
# print(np.shape(GT))


#%%
image_1 = INPT_1[0]
image_2 = INPT_1[1]

cloud_1 = Cloud[0]
cloud_2 = Cloud[1]

#%%

pos_cloud_1 = np.where(cloud_1 > 10)
pos_cloud_2 = np.where(cloud_2 > 10 )

#%%

image_1[pos_cloud_1] = 0

#%%
# np.save('D:/ImageInterpretation/lab2/first_image.npy', image_1)
# np.save('D:/ImageInterpretation/lab2/first_nir.npy', nir_1)
# np.save('D:/ImageInterpretation/lab2/first_gt1.npy', gt_1)