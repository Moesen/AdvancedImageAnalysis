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

im = skimage.io.imread(data_path + im_name).astype(np.float)

fig, ax = plt.subplots(1)
ax.imshow(im)

#%% Compute Gaussian kernels

def gauss_kernels(sigma, size=4):
    s = np.ceil(np.max([sigma*size, size]))
    x = np.arange(-s,s+1)
    x = x.reshape(x.shape + (1,))
    g = np.exp(-x**2/(2*sigma*sigma))
    g /= np.sum(g)
    dg = -x/(sigma*sigma)*g
    ddg = -1/(sigma*sigma)*g -x/(sigma*sigma)*dg
    return g, dg, ddg

sigma = 2
g, dg, ddg = gauss_kernels(sigma)

size = 4
s = np.ceil(np.max([sigma*size, size]))
x = np.arange(-s,s+1)
x = x.reshape(x.shape + (1,))

fig, ax = plt.subplots(1,3)
ax[0].plot(x,g)
ax[0].set_title('Gaussian')
ax[0].set_ylim(-np.max(g),np.max(g))
ax[1].plot(x,dg)
ax[1].set_title('1st order derivative')
ax[1].set_ylim(-np.max(g),np.max(g))
ax[2].plot(x,ddg)
ax[2].set_title('2nd order derivative')
ax[2].set_ylim(-np.max(g),np.max(g))




#%% Comptue the 2D Gaussian kernel

g2d = np.outer(g,g)
x = np.outer(np.linspace(0, g2d.shape[0], g2d.shape[0]), np.ones(g2d.shape[0]))
y = x.copy().T # transpose
z = g2d

fig, ax = plt.subplots(1)
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z,cmap='jet', edgecolor='none')


#%% 1.1 Verify the separability of the Gaussian kernel

im_g_2d = scipy.ndimage.convolve(im, g2d)
im_g_two_1d = scipy.ndimage.convolve(scipy.ndimage.convolve(im,g),g.T)


fig, ax = plt.subplots(1)
ax.imshow(im_g_2d-im_g_two_1d)

#%% 1.2 Derivative by Gaussian and central difference
# Shows that the small derivative is going to give the same result whereas 
# larger kernels will give a smoother result for the Gaussian

sigma = 0.2
g, dg, ddg = gauss_kernels(sigma)

k = np.array([[0.5,0,-0.5]]).T
im_dx_g = scipy.ndimage.convolve(scipy.ndimage.convolve(im,dg),g.T)
im_dx_c = scipy.ndimage.convolve(im,k)

fig, ax = plt.subplots(1,2)
ax[0].imshow(im_dx_g)
ax[1].imshow(im_dx_c)


#%% 1.3 Large convolution of t = 20 equal to 10 convolutions of t = 2

t = 20
sigma_20 = np.sqrt(t)
size = 5
g20 = gauss_kernels(sigma_20,size=size)[0]

im_20 = scipy.ndimage.convolve(scipy.ndimage.convolve(im,g20),g20.T)

t = 2
sigma_2 = np.sqrt(t)
g2 = gauss_kernels(sigma_2,size=size)[0]

im_2_times_10 = scipy.ndimage.convolve(scipy.ndimage.convolve(im,g2),g2.T)
for i in range(1,10):
    im_2_times_10 = scipy.ndimage.convolve(scipy.ndimage.convolve(im_2_times_10,g2),g2.T)


fig, ax = plt.subplots(1,3)
ax[0].imshow(im_20)
ax[1].imshow(im_2_times_10)
ax[2].imshow(im_20-im_2_times_10)

#%% Large derivative with t = 20 compared to Gaussian and small derivative with t = 10

t = 20
sigma_20 = np.sqrt(t)
size = 5
g20, dg20 = gauss_kernels(sigma_20,size=size)[:2]

t = 10
sigma_10 = np.sqrt(t)
g10, dg10 = gauss_kernels(sigma_10,size=size)[:2]


im_x_20 = scipy.ndimage.convolve(scipy.ndimage.convolve(im,dg20),g20.T)
im_x_10_times_2 = scipy.ndimage.convolve(scipy.ndimage.convolve(im,g10),g10.T)
im_x_10_times_2 = scipy.ndimage.convolve(scipy.ndimage.convolve(im_x_10_times_2,dg10),g10.T)

fig, ax = plt.subplots(1,3,figsize=(15,5))
pos0 = ax[0].imshow(im_x_20)
fig.colorbar(pos0, ax=ax[0], shrink=0.65)
pos1 = ax[1].imshow(im_x_10_times_2)
fig.colorbar(pos1, ax=ax[1], shrink=0.65)
pos2 = ax[2].imshow(im_x_20-im_x_10_times_2)
fig.colorbar(pos2, ax=ax[2], shrink=0.65)





