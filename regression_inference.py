# -*- coding: utf-8 -*-
import os
import time
import copy
import h5py
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import server_train_data_loader_yatao


if __name__ == '__main__':
    # root = r'C:\ETH Course\Image interpretation\Classification\Sentinel-dset\Sentinel-dset'
    root = r'C:\ETH Course\Image interpretation\Regression\Image_interpretation_both_groups\lab 2 data'
    # root = '../lab_2_data'

    trained_model_name = 'model_fcn_resnet50_best'

    # dataset and some basic variables
    dataTest = root + '/training_data.hdf5'
    test_dataset = server_train_data_loader_yatao.ImageLoader(windowsize=256, test=False, datafile=dataTest, _num_smpls=2)

    # generate test loader
    num_batches = 1
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=num_batches,
                                              num_workers=8,
                                              shuffle=True)
    print("Test size: {}".format(len(test_loader)))

    # determine whether gpu will be used in the training process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # select regression model: trained_version
    model_ft = torch.load(trained_model_name)

    # create the optimizer
    model_ft = model_ft.to(device)
    model_ft.eval()
    # transform only works for resnet
    transform = transforms.Compose([transforms.Normalize(mean=[0.406, 0.456, 0.485, 0.485],
                                                         std=[0.225, 0.224, 0.229, 0.229])])
    model_ft = model_ft.to(device)

    _rmse = 0.0
    _max = -np.inf
    _min = np.inf
    _ave = 0.0
    _pixel_num = 0
    begin_time = time.time()
    print('before inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("Infer with {} samples".format(len(test_loader)))
    for inputs, labels in test_loader:
        ##############################################################################
        # inputs = inputs.float().to(device)  # for unet and mlp
        inputs = transform(inputs.float()).to(device)  # for resnet
        ##############################################################################
        labels = labels.float().to(device)
        with torch.no_grad():
            outputs = model_ft.forward(inputs)
            ##############################################################################
            # outputs = torch.squeeze(outputs)  # for unet and mlp
            outputs = torch.squeeze(outputs['out'])  # for resnet
            ##############################################################################

            mask = labels.ge(0.)
            mask = torch.squeeze(mask)
            labels = torch.squeeze(labels)

            if mask.sum() == 0:
                continue

            # accumulative rmse
            _pixel_num += np.sum(np.array(mask.cpu()))
            _residual = np.array(outputs[mask].cpu()) - np.array(labels[mask].cpu())
            _rmse += np.sum(_residual ** 2)
            _ave += np.sum(_residual)
            _max = np.max([_max, np.max(_residual)])
            _min = np.min([_min, np.min(_residual)])

    print('Inference time: {}'.format(time.time() - begin_time))
    print('after inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    _rmse = np.sqrt(_rmse / float(_pixel_num))
    _ave = _rmse / float(_pixel_num)
    print("RMSE: {}".format(_rmse))
    print("Mean: {}".format(_ave))
    print("Max: {}".format(_max))
    print("Min: {}".format(_min))
