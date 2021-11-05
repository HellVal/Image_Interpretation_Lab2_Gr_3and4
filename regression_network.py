# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:07:44 2021

@author: paimhof
"""


#source: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

#%% import section 

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math 
import random 
from tqdm import tqdm

from numpy import vstack
from sklearn.metrics import accuracy_score

import train_data_loader
import torch


import regression_model
import torch.nn as nn
from torch import optim as op


#%% define variables

num_batches = 8
cut_size = 128
num_layers = 4


# hyperparameters
num_epocs = 1
learning_rate = 0.000001
#%%check for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# probably add .is_cuda to the code to run the complete pipeline on the GPU

#%% create train and validadtion size

trainset = train_data_loader.ImageLoader( windowsize = cut_size,test=False)

train_size = int(len(trainset)*0.01)
valid_size = len(trainset)-train_size




# TODO add the test data set 
# propbably add additional dataloader

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
# TODO in the dataloader!!



#%% define network

#model = regression_model.regressionModel(100).to(device)
#model = regression_model.UNet(4,1).to(device)
model = regression_model.UNet2().to(device)

#model = regression_model.FullyConnected(num_batches*num_layers*cut_size*cut_size,100,50,1)
model.train()


# define the optimization/Loss
criterion = nn.MSELoss()
# set optimizer 
optimizer = op.SGD(model.parameters(), lr=learning_rate, momentum=0.9)



#%%

loss_data = []

# enumerate epochs
for epoch in tqdm(range(num_epocs)):
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_loader):
        print(i)
        inputs = nn.functional.normalize(inputs)
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        yhat = torch.squeeze(yhat)
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
    
    predictions, actuals = list(), list()
    for j, (inputs_v, targets_v) in enumerate(validation_loader):
        if j >10:
            break
        # evaluate model on the validation set
        inputs_v = nn.functional.normalize(inputs_v)
        yhat_v = model(inputs_v)
        yhat_v = torch.squeeze(yhat_v)
        # numpy array
        yhat_v = yhat_v.detach().numpy()
        actual = targets_v.numpy()
        #actual = actual.reshape((len(actual),1))
        
        predictions.append(yhat)
        actuals.append(actual)
        
    # predictions, actuals = vstack(predictions), vstack(actuals)
    # acc = accuracy_score(actuals,predictions)
    
    
        
        
       # loss_data.append(loss.detach().nump())
       
   # add validation after each epoch 
   # TODO compare training loss with validation loss
   # evaluate the model
    

    
