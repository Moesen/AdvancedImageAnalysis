#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 01:25:23 2021

@author: vand
"""

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import maxflow.fastmin

#%%
path = '../../../../Data/week5/'
GT = skimage.io.imread(path + 'noise_free_circles.png') # ground truth
I = skimage.io.imread(path + 'noisy_circles.png')
mu = np.unique(GT) # instead of estimating

# converting all to float
I = I.astype(np.float)/255
GT = GT.astype(np.float)/255
mu = mu.astype(np.float)/255

beta = 0.05
D = np.stack([(I-mu[i])**2 for i in range(3)], axis=2)
V = beta - beta*np.eye(len(mu), dtype = D.dtype)

#%%
S0 = np.argmin(D, axis=2)
S = S0.copy()
maxflow.fastmin.aexpansion_grid(D, V, labels = S)

D0 = mu[S0]
D = mu[S] # denoised image using intensities from mu

fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
ax[0].imshow(GT, vmin=0, vmax=1, cmap=plt.cm.gray)
ax[0].set_title('noise-free')
ax[1].imshow(I, vmin=0, vmax=1, cmap=plt.cm.gray)
ax[1].set_title('noisy')
ax[2].imshow(D0, vmin=0, vmax=1, cmap=plt.cm.gray)
ax[2].set_title('max likelihood')
ax[3].imshow(D, vmin=0, vmax=1, cmap=plt.cm.gray)
ax[3].set_title('max posterior')

