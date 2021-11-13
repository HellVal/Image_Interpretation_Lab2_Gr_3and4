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
    
#%%

# source: https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/


from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
