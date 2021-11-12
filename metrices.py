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

def computeMetrices(predicted, gt, outfile):
    residuals = gt - predicted
    
    residuals_min = np.amin(residuals)
    residuals_max = np.amax(residuals)
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)
    residuals_median = np.median(residuals)
    residual_MAD = stats.median_absolute_deviation(residuals)
    residuals_mode = statistics.mode(residuals.astype(int))
    
    metrices = {
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
        sys.stout = f
        print('File name:', predicted)
        for k in metrices.keys():
            print(f'{k}: {metrices[k]}')
            
        sys.stdout = original_stdout
        
    print(f'Metrices saved to {outfile}')