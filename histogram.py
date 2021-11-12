# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:57:01 2021

@author: jfandre
@purpose: find good images
"""
# Loading Packages
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

# Input Parameters
root = r'P:\pf\pfshare\data\mikhailu'
files = ["\dataset_rgb_nir_train.hdf5", "\dataset_rgb_nir_test.hdf5"]
output = r"P:\pf\pfstud\II_Group3"

# Loading Data
file = root + files[1]
h5 = h5py.File(file, 'r')

# #train data
# img_names = ['INPT_1', 'INPT_2', 'INPT_3', 'INPT_4']
# nir_names = ['NIR_1', 'NIR_2', 'NIR_3', 'NIR_4']
# cld_names = ['CLD_1', 'CLD_2', 'CLD_3', 'CLD_4']

#test data
img_names = ['INPT_0', 'INPT_5']
nir_names = ['NIR_0', 'NIR_5']
cld_names = ['CLD_0', 'CLD_5']
#%% 
#check rgb

for j in range(len(img_names)): #img_names.size    len(img_names)
    img = h5[img_names[j]]
    cld = h5[cld_names[j]]
    
    for i in range(img.shape[0]):
        rgb_1 = img[i]
        cld_1 = cld[i]
        # Creating Mask
        mask_cld_1 = np.where(cld_1 <= 10)
        imgFiltered = rgb_1[mask_cld_1]
        
        
        # Create plot
        fig, axs = plt.subplots(3,1, figsize = (30,15))
        string = "{}: {}".format(img_names[j], i)
        fig.suptitle(string)
        
        for k in range(3):
            axs[k].hist(imgFiltered[:,k], 1000)
        
        plt.show()

#%%        
for j in range(len(nir_names)): #img_names.size    len(img_names)
    img = h5[nir_names[j]]
    cld = h5[cld_names[j]]
    
    for i in range(img.shape[0]):
        rgb_1 = img[i]
        cld_1 = cld[i]
        # Creating Mask
        mask_cld_1 = np.where(cld_1 <= 10)
        imgFiltered = rgb_1[mask_cld_1]
        
        
        # Create plot
        plt.figure(figsize=(15,10))
        plt.hist(imgFiltered, 1000)
        string = "{}: {}".format(nir_names[j], i)
        plt.title(string)
        
        plt.show()