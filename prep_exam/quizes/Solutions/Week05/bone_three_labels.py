import skimage.io
import matplotlib.pyplot as plt
import maxflow.fastmin
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


#%% Inspect the image and the histogram
path = '../../../../Data/week5/'
I = skimage.io.imread(path + 'V12_10X_x502.png').astype(float)/(2**16-1)
fig, ax = plt.subplots()
ax.imshow(I, cmap=plt.cm.gray)

edges = np.linspace(0, 1, 257)
fig, ax = plt.subplots()
ax.hist(I.ravel(), edges)
ax.set_xlabel('pixel values')
ax.set_ylabel('count')
ax.set_title('intensity histogram')
#%% Define likelihood
mu = np.array([0.40, 0.46, 0.71])
U = np.stack([(I-mu[i])**2 for i in range(len(mu))],axis=2)
S0 = np.argmin(U,axis=2)

fig, ax = plt.subplots()
ax.imshow(S0)
ax.set_title('max likelihood')

#%% Define prior and solve
beta  = 0.02
B = beta - beta*np.eye(len(mu))
#  solving
S = S0.copy()
maxflow.fastmin.aexpansion_grid(U, B, labels = S)


fig, ax = plt.subplots()
ax.imshow(S)
ax.set_title('max posterior')

fig, ax = plt.subplots()
segmentation_histogram(ax, I, S, edges=edges)
ax.set_aspect(1./ax.get_data_ratio())
ax.set_xlabel('pixel values')
ax.set_ylabel('count')
ax.set_title('segmentation histogram')


