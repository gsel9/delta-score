import numpy as np 
import networkx as nx

from sklearn.metrics import jaccard_score


def check_adjacency(A):
    # Sanity checks.

    assert np.isclose(np.sum(np.diag(A)), 0)

    assert np.array_equal(A, A.T)

    return A


def intial_adjacency(n, p, seed):
    
    # Initialize with random adjancey.
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=False)
    return check_adjacency(nx.to_numpy_array(G))


def check_iterable(N_i):
    """Check the data structure of a neighbourhood."""
    
    if isinstance(N_i, (int, float)):
        N_i = np.array([int(N_i)])
        
    if not isinstance(N_i, np.ndarray):
        N_i = np.array(N_i)
        
    if np.ndim(N_i) < 1:
        N_i = np.expand_dims(N_i, axis=0)
        
    return N_i


def interpolation_region_mask(Y):
    
    O = np.zeros_like(Y)
    for i, y in enumerate(Y):
        
        t_start = np.argmax(Y[i] != 0)
        t_end = np.argmax(np.cumsum(Y[i])) + 1
        
        O[i, t_start:t_end] = 1
        
    return O


def solve_M_hat(L, M, M_star, beta, gamma):
    """Solves \min_{\hat{M}} | M - \hat{M} |_F^2 + \gamma tr(\hat{M}^\top L \hat{M})."""
    
    return np.array(np.linalg.inv(np.eye(M.shape[0]) + gamma * L) @ (beta * M + M_star * (1 - beta)))