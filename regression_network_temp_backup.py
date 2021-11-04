# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:07:44 2021

@author: paimhof
"""

#%% import section 

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math 
import random 
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#%%
import torch 
from dataset_windows import SatelliteSet

#%% define variables
windowsize = 1024.0
img_size = 10980.0

#%% load data section 
# path = 'P:/pf/pfshare/data/mikhailu/'
# name_train = 'dataset_rgb_nir_train.hdf5'
# name_test = 'dataset_rgb_nir_test.hdf5'


# # load training data
# dset_train = h5py.File(path+name_train,"r")

# # load test data 
# dset_test = h5py.File(path+name_test,"r")


# #input image 1
# INPT_1 = dset_train["INPT_1"]
# NIR_1 = dset_train["NIR_1"]
# CLOUD_1 = dset_train["CLD_1"]
# # groundtruth
# GT = dset_train["GT"]


# print(np.shape(INPT_1))
# print(np.shape(NIR_1))
# print(np.shape(CLOUD_1))
# print(np.shape(GT))





#%% create data

# check which windwo has to be croped
temp = img_size%windowsize 
size_last_b = 0

if temp == 0:
    b = img_size/windowsize
else:
    b = math.floor(img_size/windowsize)
    size_last_b = temp
    
x_coord =  np.arange(0,b*windowsize,windowsize)
y_coord =  np.arange(0,b*windowsize,windowsize)

# create grid
xx , yy = np.meshgrid(x_coord,y_coord)



#%% create simple test data take 10 patches from the first image

start = 0
end = img_size-windowsize
n = 20

coord = [random.randint(start, end) for _ in range(n)]

#create empty input array
test_input = np.zeros((int(n*np.square(windowsize)),4))
test_groundtruth = np.zeros((int(n*np.square(windowsize)),1))

test_image = 2

# insert data
for i in range(1,n+1):
    test_input[int((i-1)*np.square(windowsize)):int(i*np.square(windowsize)),:3] = np.reshape(INPT_1[test_image,coord[i-1]:int(coord[i-1]+windowsize),coord[i-1]:int(coord[i-1]+windowsize),:],(int(np.square(windowsize)),3))
    test_input[int((i-1)*np.square(windowsize)):int(i*np.square(windowsize)),3] = np.reshape(NIR_1[test_image,coord[i-1]:int(coord[i-1]+windowsize),coord[i-1]:int(coord[i-1]+windowsize)],(int(np.square(windowsize))))
    test_groundtruth[int((i-1)*np.square(windowsize)):int(i*np.square(windowsize)),0] = np.reshape(GT[test_image,coord[i-1]:int(coord[i-1]+windowsize),coord[i-1]:int(coord[i-1]+windowsize)],(int((np.square(windowsize)))))



#%% normalize data






#%% define network
def build_and_compile_model():
  model = keras.Sequential([
      #norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
  return model



# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
test_model = build_and_compile_model()

#%%
callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]



#%% train network

# dataloader



history = test_model.fit(test_input, test_groundtruth, validation_split=0.2, verbose=0, epochs=100, callbacks=callbacks)



#%% summary of network

test_model.summary()

#%% Analysis of Network
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


#%% function to plot the loss function
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  
  
#%% plot loss
plot_loss(history)

#%% predict with the model
print(test_model.predict(test_input[:100,:]))