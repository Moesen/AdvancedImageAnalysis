#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 03:15:19 2021

@author: vand
"""

import skimage.io
import matplotlib.pyplot as plt
import maxflow
import numpy as np

def segmentation_histogram(ax, I, S, edges=None):
    '''
    Histogram for data and each segmentation label.
    '''
    if edges is None:
        edges = np.linspace(I.min(), I.max(), 100)
    ax.hist(I.ravel(), bins=edges, color = 'k')
    centers = 0.5*(edges[:-1] + edges[1:]);
    for k in range(S.max()+1):
        ax.plot(centers, np.histogram(I[S==k].ravel(), edges)[0])


# noisy image
path = '../../../../Data/week5/'
I = skimage.io.imread(path + 'DTU_noisy.png').astype(float)/255

# MRF parameters
beta  = 0.1
mu = [90/255, 170/255]

# Setting up graph with internal and external edges
g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(I.shape)
g.add_grid_edges(nodeids, beta)
g.add_grid_tedges(nodeids, (I-mu[1])**2, (I-mu[0])**2)

#  Graph cut
g.maxflow()
S = g.get_grid_segments(nodeids)

# Visualization
fig, ax = plt.subplots(1, 3)
ax[0].imshow(I, vmin=0, vmax=1, cmap=plt.cm.gray)
ax[0].set_title('Noisy image')
ax[1].imshow(S)
ax[1].set_title('Segmented')
segmentation_histogram(ax[2], I, S, edges=None)
ax[2].set_aspect(1./ax[2].get_data_ratio())
ax[2].set_title('Segmentation histogram')

