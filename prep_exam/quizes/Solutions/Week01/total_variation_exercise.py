#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:31:24 2020

@author: vand
"""

# optional exercise 1.1.4
import skimage.io
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def total_variation(I):
    dx,dy = np.gradient(I)
    v = np.sum(abs(dx))+np.sum(abs(dy))
    return(v)

path = '../../../../Data/week1/'
I = skimage.io.imread(path + 'fibres_xcth.png')

I = I.astype('float')/(2**16-1)
I = I[300:700,300:700]

v = total_variation(I)
Is = scipy.ndimage.gaussian_filter(I, sigma=5, truncate=3, mode='nearest')
vs = total_variation(Is)

fig, ax = plt.subplots(1,2)
ax[0].imshow(I, cmap='gray')
ax[0].set_title(f'total variation is {v}')
ax[1].imshow(Is, cmap='gray')
ax[1].set_title(f'total variation is {vs}')


