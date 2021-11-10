# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:59:01 2021

@author: paimhof
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn



#%% Unet 2

def double_conv(in_c, out_c):
  conv = nn.Sequential(
    nn.Conv2d(in_c,out_c,3,padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_c, out_c,3,padding=1),
    nn.ReLU(inplace=True)
  )
  return conv

def crop_img(input_tensor,target_tensor):
  target_size = target_tensor.shape[2]
  input_size = input_tensor.shape[2]
  crop = fn.center_crop(input_tensor, output_size=target_size)
  return crop


class UNet2(nn.Module):
  def __init__(self):
    super(UNet2,self).__init__()

    self.max_pool_2x2 = nn.MaxPool2d(2,2)
    self.down_conv_1 = double_conv(4,64)
    self.down_conv_2 = double_conv(64,128)
    self.down_conv_3 = double_conv(128,256)
    self.down_conv_4= double_conv(256,512)
    self.down_conv_5 = double_conv(512,1024)

    self.up_trans_1 = nn.ConvTranspose2d(1024,512,2,2)
    self.up_conv_1 = double_conv(1024,512)

    self.up_trans_2 = nn.ConvTranspose2d(512,256,2,2)
    self.up_conv_2 = double_conv(512,256)
    
    self.up_trans_3 = nn.ConvTranspose2d(256,128,2,2)
    self.up_conv_3 = double_conv(256,128)

    self.up_trans_4 = nn.ConvTranspose2d(128,64,2,2)
    self.up_conv_4 = double_conv(128,64)

    self.regressor = nn.Conv2d(64,1,1)

  def forward(self,image):
    #encoder
    x1 = self.down_conv_1(image) #
    x2 = self.max_pool_2x2(x1)
    x3 = self.down_conv_2(x2)  #
    x4 = self.max_pool_2x2(x3)
    x5 = self.down_conv_3(x4)   #
    x6 = self.max_pool_2x2(x5)
    x7 = self.down_conv_4(x6)  #
    x8 = self.max_pool_2x2(x7)
    x9 = self.down_conv_5(x8)


    #decoder
    x = self.up_trans_1(x9)
    y = crop_img(x7,x)
    x = self.up_conv_1(torch.cat([y,x],1))

    x = self.up_trans_2(x)
    y = crop_img(x5,x)
    x = self.up_conv_2(torch.cat([y,x],1))

    x = self.up_trans_3(x)
    y = crop_img(x3,x)
    x = self.up_conv_3(torch.cat([y,x],1))

    x = self.up_trans_4(x)
    y = crop_img(x1,x)
    x = self.up_conv_4(torch.cat([y,x],1))

    x = self.regressor(x)
    return x

#%%
class MLP(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(4, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 1)
        )


    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)
