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
from tqdm import tqdm

from torchinfo import summary


import train_data_loader
import torch


import regression_model
import torch.nn as nn
from torch import optim as op


#%% define variables

img_size = 10980.0



num_batches = 8
cut_size = 128
#%%check for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#%% create train and validadtion size

trainset = train_data_loader.ImageLoader( windowsize = cut_size,test=False)


train_size = int(len(trainset)*0.05)
valid_size = len(trainset)-train_size



#%% create dataloader
train_dataset, validation_dataset = torch.utils.data.random_split(trainset, [train_size, valid_size])
        
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=num_batches,
                                            num_workers=0,
                                            shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                            batch_size=num_batches,
                                            num_workers=0,
                                            shuffle=True)

#%% normalize data
# TODO in the loader!!



#%% define network

#model = regression_model.regressionModel(100).to(device)
#model = regression_model.UNet(4,1).to(device)
model = regression_model.UNet2().to(device)
model.train()


# define the optimization
criterion = nn.MSELoss()
optimizer = op.SGD(model.parameters(), lr=0.01, momentum=0.9)


#%%
# enumerate epochs
for epoch in tqdm(range(2)):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_loader):
        print(i)
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()