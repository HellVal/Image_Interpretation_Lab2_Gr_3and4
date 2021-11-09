# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:08:04 2021

@author: paimhof
"""


import h5py
import numpy as np
from torchvision.datasets.vision import VisionDataset


class ImageLoader(VisionDataset):
    def __init__(self, windowsize=128, test=False,datafile=''):
        self.wsize = windowsize
        super().__init__(None)
        self.num_smpls, self.sh_x, self.sh_y = 4, 10980, 10980  # size of each image

        self.pad_x = (self.sh_x - (self.sh_x % self.wsize))
        self.pad_y = (self.sh_y - (self.sh_y % self.wsize))
        self.sh_x = self.pad_x + self.wsize
        self.sh_y = self.pad_y + self.wsize
        self.num_windows = self.num_smpls * self.sh_x / self.wsize * self.sh_y / self.wsize
        self.num_windows = int(self.num_windows)
        self.dfile = datafile
        self.has_data = False

    # ugly fix for working with windows
    # Windows cannot pass the h5 file to sub-processes, so each process must access the file itself.
    def load_data1(self):
        h5 = h5py.File(self.dfile, 'r')
        self.RGB = h5["INPT"]
        self.GT = h5["GT"]
        self.has_data = True

    def __getitem__(self, index):
        if not self.has_data:
            self.load_data1()

        # Returns a data sample from the dataset.
        # determine where to crop a window from all images (no overlap)
        m = index * self.wsize % self.sh_x  # iterate from left to right
        # increase row by windows size everytime m increases
        n = (int(np.floor(index * self.wsize / self.sh_x)) * self.wsize) % self.sh_x
        # determine which batch to use
        b = (index * self.wsize * self.wsize // (self.sh_x * self.sh_y)) % self.num_smpls

        # crop all data at the previously determined position
        # RGB_sample_1 = self.RGB[n:n + self.wsize, m:m + self.wsize]
        # GT_sample_1 = self.GT[n:n + self.wsize, m:m + self.wsize]
        RGB_sample = self.RGB[b, n:n + self.wsize, m:m + self.wsize]
        GT_sample = self.GT[b, n:n + self.wsize, m:m + self.wsize]
        # print(b, RGB_sample.shape, GT_sample.shape, RGB_sample_1.shape, GT_sample_1.shape)

        RGB_sample = np.asarray(RGB_sample, np.float32) / (2 ** 8 - 1)
        X_sample = RGB_sample

        # pad the data if size does not match
        sh_x, sh_y = np.shape(GT_sample)
        pad_x, pad_y = 0, 0
        if sh_x < self.wsize:
            pad_x = self.wsize - sh_x
        if sh_y < self.wsize:
            pad_y = self.wsize - sh_y

        x_sample = np.pad(X_sample, [[0, pad_x], [0, pad_y], [0, 0]])
        gt_sample = np.pad(GT_sample, [[0, pad_x], [0, pad_y]], 'constant',
                           constant_values=[0])  # pad with 0 to mark absence of data

        # pytorch wants the data channel first - you might have to change this
        x_sample = np.transpose(x_sample, (2, 0, 1))

        # mask to set rgb->nan as 0 and set gt<0 as 0
        x_sample_mask = np.isnan(x_sample)
        x_sample_mask_all = np.logical_or(np.logical_or(np.logical_or(x_sample_mask[0], x_sample_mask[1]),
                                                        x_sample_mask[2]), x_sample_mask[3])
        gt_sample_mask = gt_sample < 0
        data_sample_mask = np.logical_or(x_sample_mask_all, gt_sample_mask)

        # set mask=false as 0
        for i in range(len(x_sample)):
            x_sample[i][data_sample_mask] = 0
        gt_sample[data_sample_mask] = 0

        return np.asarray(x_sample), gt_sample

    def __len__(self):
        return self.num_windows