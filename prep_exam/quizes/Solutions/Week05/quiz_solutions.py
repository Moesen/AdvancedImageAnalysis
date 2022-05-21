import numpy as np
            
def prior_energy(S, beta):
    return beta * ((np.diff(S, axis=0)!=0).sum() + (np.diff(S, axis=1)!=0).sum())   
    
def likelihood_energy(S, I, mu):
    return ((mu[S] - I)**2).sum()

I = np.array([[1, 2, 6, 4, 10,  8],
              [4, 1, 3, 5, 9, 6],
              [5, 2, 3, 5, 4, 7]])
mu = np.array([2, 5, 10])
beta = 10

#%% A
U = np.stack([I-mu[0], I-mu[1], I-mu[2]], axis=2)**2
S0 = np.argmin(U, axis=2)
prior_noisy = prior_energy(S0, beta)
print(f'prior_noiy {prior_noisy}')

#%% B
S = np.tile([0,0,1,1,2,2], (3,1)).astype(np.int)
likelihood_stripes = likelihood_energy(S, I, mu);
print(f'likelihood_stripes {likelihood_stripes}')

#%% C
S_MAP = np.array([[0, 0, 1, 1, 2, 2],
                  [0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1]])
posterior_small  = prior_energy(S_MAP, beta) + likelihood_energy(S_MAP, I, mu)
print(f'posterior_small {posterior_small}')

#%% another realization with the same posterior
S_test = np.minimum(S_MAP, 1)
posterior_test  = prior_energy(S_test, beta) + likelihood_energy(S_test, I, mu)
print(f'posterior_test {posterior_test}')

#%% C computing MAP using graph cuts
import maxflow.fastmin
S_GC = S0.copy()
maxflow.fastmin.aexpansion_grid(U, beta - 
                            beta*np.eye(3, 3, dtype=U.dtype), labels = S_GC)
posterior_GC  = prior_energy(S_GC, beta) + likelihood_energy(S_GC, I, mu)
print(f'posterior_GC {posterior_GC}')

