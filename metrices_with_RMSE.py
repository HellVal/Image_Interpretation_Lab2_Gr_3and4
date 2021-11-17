# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:00:11 2021
@author: jfandre
@purpose: Computing of RMSE and mean
"""

import numpy as np
import statistics
from scipy import stats
import os
import sys

def computeMetrices(predicted, gt, outfile,filename):
    residuals = gt - predicted
    
    residuals_min = np.amin(residuals)
    residuals_max = np.amax(residuals)
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)
    rmse = np.sqrt(((-1 * residuals) ** 2).mean())
    residuals_median = np.median(residuals)
    residual_MAD = stats.median_abs_deviation(residuals) # Median absolut deviation from median
    residuals_mode = statistics.mode(residuals.astype(int))
    
    metrices = {
        'RMSE': rmse,
        'Minimum': residuals_min,
        'Maximum': residuals_max,
        'Mean': residuals_mean,
        'Standard deviation': residuals_std,
        'Median': residuals_median,
        'MAD': residual_MAD,
        'Mode': residuals_mode
        }
    
    if os.path.exists(outfile):
        os.remove(outfile)
    
    original_stdout = sys.stdout
    with open(outfile,'w') as f:
        sys.stdout = f
        print('File name:', filename)
        for k in metrices.keys():
            print(f'{k}: {metrices[k]}')
            
        sys.stdout = original_stdout
        
    print(f'Metrices saved to {outfile}')
    
a = np.array([1,2,3,5,6,7,8,9])
b = np.array([2,3,3,4,5,6,6,9])
output = r'C:/Users/Josianne/Desktop/ETH/Semester 3/Image Interpretation/Lab 2/test.txt'
filename = 'test'
test = computeMetrices(a,b,output, filename)