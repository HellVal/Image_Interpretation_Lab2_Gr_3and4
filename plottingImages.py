# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:16:10 2021

@author: jfandre
"""
# Loading Packages
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

# Input Parameters
# root = r'P:\pf\pfshare\data\mikhailu'
# files = ["\dataset_rgb_nir_train.hdf5", "\dataset_rgb_nir_test.hdf5"]
# output = r"P:\pf\pfstud\II_Group3"


root = r'P:\pf\pfstud\II_Group3'
files = ["\dataset_rgb_nir_train.hdf5", "\dataset_rgb_nir_test.hdf5"]
output = r"P:\pf\pfstud\II_Group3"
# Loading Data
file = root + files[0]
h5 = h5py.File(file, 'r')
image = h5['INPT_1'][:,:,:3]/2500

# image = h5['INPT_1'][3,:,:,:]/2500
# clouds = h5['CLD_2'][3,:,:]
# nir = h5['NIR_2'][3,:,:]

plt.imshow(image)
plt.show()

# plt.imshow(clouds)
# plt.show()

# plt.imshow(nir)
# plt.show()
