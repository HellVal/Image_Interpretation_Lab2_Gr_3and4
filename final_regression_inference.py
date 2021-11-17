# -*- coding: utf-8 -*-
"""
Created on Wed Nov 3

@author: Yatao Zhang
"""

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
import final_train_data_loader_fcn_resnet
from random import randint
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    root = '../lab_2_data'
    trained_model_name = './model_fcn_resnet50_best'

    # dataset and some basic variables
    dataTest = root + '/test_data.hdf5'
    test_dataset = final_train_data_loader_fcn_resnet.ImageLoader(windowsize=256, test=False, datafile=dataTest, _num_smpls=2)

    # generate test loader
    num_batches = 4
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=num_batches,
                                              num_workers=8,
                                              shuffle=False)
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
    _mae = 0.0
    _max = -np.inf
    _min = np.inf
    _ave = 0.0
    _pixel_num = 0
    begin_time = time.time()
    print('before inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("Infer with {} samples".format(len(test_loader)))
    flag = 0
    test_counter = 0
    for inputs, labels in test_loader:
        flag += 1
        if flag % 100 == 0:
            print("data-{}".format(flag))
        ##############################################################################
        # inputs = inputs.float().to(device)  # for unet and mlp
        inputs_vis = inputs.float().to(device)
        inputs = transform(inputs.float()).to(device)  # for resnet
        ##############################################################################
        labels = labels.float().to(device)
        with torch.no_grad():
            outputs = model_ft.forward(inputs)
            ##############################################################################
            outputs = outputs['out']  # for resnet
            outputs_masked = torch.squeeze(outputs)  # for unet and mlp
            ##############################################################################

            mask = labels.ge(0.)
            mask = torch.squeeze(mask)
            labels_masked = torch.squeeze(labels)

            if mask.sum() == 0:
                continue

            # accumulative rmse
            _pixel_num += np.sum(np.array(mask.cpu()))
            _residual = np.array(outputs_masked[mask].cpu()) - np.array(labels_masked[mask].cpu())
            _rmse += np.sum(_residual ** 2)
            _mae += np.sum(np.abs(_residual))
            _ave += np.sum(_residual)
            _max = np.max([_max, np.max(_residual)])
            _min = np.min([_min, np.min(_residual)])

            test_counter += 1
            if test_counter % 50 == 0:
                s_inputs = inputs_vis.shape
                # random_number = randint(0, s_inputs[0] - 1)
                random_number = 0
                plot_inputs = inputs_vis.cpu().detach().numpy()
                plot_inputs = np.transpose(plot_inputs, (0, 2, 3, 1))
                plot_labels = labels.cpu().detach().numpy()
                plot_outputs = outputs.cpu().detach().numpy()

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(plot_inputs[random_number, :, :, :3])
                ax1.title.set_text('input data RGB')
                ax2.imshow(plot_labels[random_number, :, :])
                ax2.title.set_text('labels')

                ax3.imshow(plot_outputs[random_number, 0, :, :])
                ax3.title.set_text('output')
                fig.savefig('./test_output/' + str(test_counter) + '.png')
                plt.close()

    print('Inference time: {}'.format(time.time() - begin_time))
    print('after inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    _rmse = np.sqrt(_rmse / float(_pixel_num))
    _mae = _mae / float(_pixel_num)
    _ave = _ave / float(_pixel_num)
    print("RMSE: {}".format(_rmse))
    print("MAE: {}".format(_mae))
    print("Mean: {}".format(_ave))
    print("Max: {}".format(_max))
    print("Min: {}".format(_min))

    f_error = open('error_result.txt', 'w')
    f_error.write("RMSE: {}\n".format(_rmse))
    f_error.write("MAE: {}\n".format(_mae))
    f_error.write("Mean: {}\n".format(_ave))
    f_error.write("Max: {}\n".format(_max))
    f_error.write("Min: {}\n".format(_min))
    f_error.close()
