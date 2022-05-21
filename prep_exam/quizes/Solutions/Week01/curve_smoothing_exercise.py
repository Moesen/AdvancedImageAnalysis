#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:26:02 2020

@author: vand
"""

# Curve smoothing, exercise 1.1.3

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def regularization_matrix(N, alpha, beta):
    """An NxN matrix for imposing elasticity and rigidity to snakes.
    Arguments: alpha is weigth for second derivative (elasticity),
    beta is weigth for (-)fourth derivative (rigidity)."""
    column = np.zeros(N)
    column[[-2,-1,0,1,2]] = alpha*np.array([0,1,-2,1,0]) + beta*np.array([-1,4,-6,4,-1])
    A = scipy.linalg.toeplitz(column)
    return(scipy.linalg.inv(np.eye(N)-A))

def regularization_matrix_version2(N, alpha, beta):
    d = alpha*np.array([-2, 1, 0, 0]) + beta*np.array([-6, 4, -1, 0])
    D = np.fromfunction(lambda i,j: np.minimum((i-j)%N,(j-i)%N), (N,N), dtype=np.int)
    A = d[np.minimum(D,len(d)-1)]
    return(scipy.linalg.inv(np.eye(N)-A))


path = '../../../../Data/week1/curves/'

X_smooth = np.loadtxt(path + 'dino.txt')
X_noisy = np.loadtxt(path + 'dino_noisy.txt')
N = X_noisy.shape[0]

closed_ind = np.r_[np.arange(N),0] # for easy plotting a closed snake

#%%
fig, ax = plt.subplots()
ax.plot(X_smooth[closed_ind,0],X_smooth[closed_ind,1],'g')
ax.plot(X_noisy[closed_ind,0],X_noisy[closed_ind,1],'r')
ax.legend(('ground truth','noisy'),loc=0)
ax.axis('equal')

#%%
# Explicit smoothing, comparing small lambda, big lambda and many iterations.
# Instead of looping, iterations implemented as matrix power.

lambda_small = 0.25;
lambda_big = 1;
nr_iters = 100;

off_diag = np.diag(np.ones(N-1),-1) + np.diag([1],N-1)
L = -2*np.diag(np.ones(N)) + off_diag + off_diag.T
smoothed_small = np.matmul(lambda_small*L+np.eye(N),X_noisy)
smoothed_big = np.matmul(lambda_big*L+np.eye(N),X_noisy)
smoothed_many = np.matmul(np.linalg.matrix_power(lambda_small*L+np.eye(N),nr_iters),X_noisy)

fig, ax = plt.subplots()
ax.plot(smoothed_small[closed_ind,0],smoothed_small[closed_ind,1],'r')
ax.plot(smoothed_big[closed_ind,0],smoothed_big[closed_ind,1],'k')
ax.plot(smoothed_many[closed_ind,0],smoothed_many[closed_ind,1],'c')
ax.legend((f'explicit 1 iteration lambda {lambda_small}',f'explicit 1 iteration lambda {lambda_big}',
           f'explicit {nr_iters} iterations lambda {lambda_small}'),loc=0)
ax.axis('equal')
#%%
# Implicit smoothing with both alpha and beta

alpha = 10;
beta = 10;

smoothed_a = np.matmul(regularization_matrix(N,alpha,0),X_noisy)
smoothed_b = np.matmul(regularization_matrix(N,0,beta),X_noisy)

fig, ax = plt.subplots()
ax.plot(smoothed_a[closed_ind,0],smoothed_a[closed_ind,1],'b')
ax.plot(smoothed_b[closed_ind,0],smoothed_b[closed_ind,1],'m')
ax.legend((f'implicit alpha {alpha}', f'implicit beta {beta}'),loc=0)
ax.axis('equal')


#%% SOLVING QUIZ 2021
# Takes in X_noisy
import numpy as np
import matplotlib.pyplot as plt

path = '../../../../Data/week1/curves/'
X_noisy = np.loadtxt(path + 'dino_noisy.txt')
N = X_noisy.shape[0]
closed_ind = np.r_[np.arange(N),0] # for easy plotting a closed snake


def curve_length(X):
    d = X-np.roll(X, shift=1, axis=0)
    d = (d**2).sum(axis=1)
    d = (np.sqrt(d)).sum()
    return(d)

a = np.array([-2, 1, 0]) 
D = np.fromfunction(lambda i,j: np.minimum((i-j)%N,(j-i)%N), (N,N), dtype=np.int)
L = a[np.minimum(D,len(a)-1)]
X_solution = np.matmul(0.25*L+np.eye(N),X_noisy)

fig, ax = plt.subplots()
ax.plot(X_noisy[closed_ind,0], X_noisy[closed_ind,1],'r')
ax.plot(X_solution[closed_ind,0], X_solution[closed_ind,1],'b--')
ax.set_title(f'noisy:{curve_length(X_noisy):.5g}, smoothed:{curve_length(X_solution):.5g}')
ax.axis('equal')

print(curve_length(X_solution))    
#%%

    

    
    

