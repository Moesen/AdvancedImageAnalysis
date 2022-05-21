#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:17:42 2021

@author: abda
"""

import numpy as np
import scipy.ndimage
import skimage.io
import matplotlib.pyplot as plt


#%% Read in image data
data_path = '../../../../Data/week1/'
im_name = 'fibres_xcth.png'

I_noisy = skimage.io.imread('../../../../Data/week1/noisy_number.png').astype(np.float)
sigma = 15;
I_smoothed = scipy.ndimage.gaussian_filter(I_noisy, sigma, mode='nearest')

fig, ax = plt.subplots()
ax.imshow(I_smoothed)
