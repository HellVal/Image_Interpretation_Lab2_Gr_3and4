import h5py
import matplotlib.pyplot as plt
import numpy as np


path = 'P:/pf/pfstud/II_Group3/'

train_dset = 'dataset_rgb_nir_train.hdf5'
test_dset = 'dataset_rgb_nir_test.hdf5'



h5 = h5py.File(path+test_dset, 'r')

print(h5.keys())

#%%
img1h5 = h5["INPT_0"]
img2h5 = h5["INPT_5"]
# img3h5 = h5["INPT_3"]
# img4h5 = h5["INPT_4"]
GTh5 = h5["GT"]

img = np.zeros((2,10980,10980,4))
img[0,:,:,:] = img1h5[:,:,:]
img[1,:,:,:] = img2h5[:,:,:]
# img[2,:,:,:] = img3h5[:,:,:]
# img[3,:,:,:] = img4h5[:,:,:]



#%% create new h5 file

f = h5py.File(path+"final_data/test_data.hdf5","w");


f.create_dataset("INPT", data = img)
f.create_dataset("GT", data = GTh5[:,:])