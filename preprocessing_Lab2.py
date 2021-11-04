# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Loading Packages
import h5py
import numpy as np
import os

# Input Parameters
root = r'P:\pf\pfshare\data\mikhailu'
files = ["\dataset_rgb_nir_train.hdf5", "\dataset_rgb_nir_test.hdf5"]
output = r"P:\pf\pfstud\II_Group3"

# Loading Data
file = root + files[0]
h5 = h5py.File(file, 'r')
#(h5.keys())

img_names = ['INPT_1', 'INPT_2', 'INPT_3', 'INPT_4']
nir_names = ['NIR_1', 'NIR_2', 'NIR_3', 'NIR_4']
cld_names = ['CLD_1', 'CLD_2', 'CLD_3', 'CLD_4']

file_name = output + files[0]
hf = h5py.File(file_name, 'w')


for j in range(1): #img_names.size
    img = h5[img_names[j]]
    nir = h5[nir_names[j]]
    cld = h5[cld_names[j]]
    

    img_new = np.zeros((img[1].shape[0],img[1].shape[1] ,4))
    mean = np.zeros((img[1].shape[0],img[1].shape[1] ,4))
    
    for i in range(img.shape[0]):
        # Taking images out
        
        rgb_1 = img[i]
        nir_1 = nir[i]
        img_1 = np.dstack((rgb_1,nir_1))
        
        cld_1 = cld[i]
        
        # Creating Mask
        mask_cld_1 = np.where(cld_1 <= 10)
        
        # Mean over all Pixels, where the pixels with to much cloud coverage get excluded
        img_temp = np.zeros((img[1].shape[0],img[1].shape[1] ,4))
        img_temp[mask_cld_1] = img_1[mask_cld_1]
        img_new = img_new + img_temp
        
        
        mean[mask_cld_1] += 1
    
    
    img_mean = img_new / mean
    hf.create_dataset(img_names[j], data = img_mean)


# Adding the ground truth
hf.create_dataset('GT', data = h5['GT'])

hf.close()
