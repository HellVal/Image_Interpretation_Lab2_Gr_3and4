# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:12:47 2021

@author: paimhof
"""

# good tutorial: https://www.tensorflow.org/tutorials/keras/regression






import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




print(tf.__version__)



#%% load data
path = 'P:/pf/pfshare/data/mikhailu/'
training_dset = 'dataset_rgb_nir_train.hdf5'

dset_train = h5py.File(path+training_dset,"r")


#input image 1
INPT_1 = dset_train["INPT_1"]
NIR_1 = dset_train["NIR_1"]
CLOUD_1 = dset_train["CLD_1"]
# groundtruth
GT = dset_train["GT"]


print(np.shape(INPT_1))
print(np.shape(NIR_1))
print(np.shape(CLOUD_1))
print(np.shape(GT))

#%% get first image

windowsize = 1024
index = 5000

test_img = INPT_1[0,index:index+windowsize,index:index+windowsize,:]
test_nir = NIR_1[0,index:index+windowsize,index:index+windowsize]
test_cloud = CLOUD_1[0,index:index+windowsize,index:index+windowsize]
test_gt = GT[0,index:index+windowsize,index:index+windowsize]


#%% print test image
  

plt.imshow(test_img)
plt.show()

plt.imshow(test_cloud)
plt.show()


#%% print shape of test image

print(np.shape(test_img))
print(np.shape(test_nir))


test_input = np.zeros((windowsize*windowsize,4))
gt_input = np.zeros((windowsize*windowsize,1))
#%% create test input
test_input[:,0] = np.reshape(test_img[:,:,0],(windowsize*windowsize))
test_input[:,1] = np.reshape(test_img[:,:,1],(windowsize*windowsize))
test_input[:,2] = np.reshape(test_img[:,:,2],(windowsize*windowsize))
test_input[:,3] = np.reshape(test_nir[:,:],(windowsize*windowsize))


gt_input[:,0] = np.reshape(test_gt[:,:],(windowsize*windowsize))
#%% normalization 


normalizer = tf.keras.layers.LayerNormalization()
normalized_input = normalizer(test_input)



#%%

linear_model = tf.keras.Sequential([
    layers.Dense(units=1)
    ])





#%%
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
plot_loss(history)


#%%
def build_and_compile_model():
  model = keras.Sequential([
      #norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
  return model

#%%

test_model = build_and_compile_model()

#%%
history = test_model.fit(
    test_input,
    gt_input,
    validation_split=0.2,
    verbose=0, epochs=5)

#%%
plot_loss(history)