import numpy as np 

from scipy import signal 
from sklearn.linear_model import LinearRegression 

try:
    from .utils import check_iterable
except: 
    from utils import check_iterable 

import numba 


@numba.jit
def shift(x, k, fill_value=0):
    
    x_shifted = np.empty_like(x)
    
    if k > 0:
        x_shifted[:k] = fill_value
        x_shifted[k:] = x[:-k]
    
    elif k < 0:
        x_shifted[k:] = fill_value
        x_shifted[:k] = x[-k:]
    
    else:
        x_shifted[:] = x
        
    return x_shifted


def cross_correlation(i, N_i, M):
    """
    Args:
        i: Index of node i.
        N_i: Indicies for the neighbours of i.
        M: Graph signals.
        eps: Profile correlation treshold.
        
    Note: 
        * Using `full` mode potentially causing profiles to be truncated. 
          Since females are born at different times, some profiles may match
          others that originally started before the female was born. 
    """
    
    # NB: Need N_i to contain the actual neighbour indicies.
    N_i = check_iterable(N_i)
    
    tau_i_star, N_i_star, C_i_star = [], [], []
    for k, m_j in enumerate(M[N_i]):

        c = signal.correlate(m_j, M[i])
        c_max = np.max(c) / sum(c != 0)

        # Retain only the neighbours sufficiently correlated.
        #if c_max >= eps:

        tau_i_star.append(np.argmax(c) + 1 - len(m_j))
            
        N_i_star.append(N_i[k])
        C_i_star.append(c_max)

    return tau_i_star, N_i_star, C_i_star


def overlaping_neighbours(M, i):
    
    # Exclude profile i.
    candidates = np.concatenate([np.arange(i), np.arange(i + 1, M.shape[0])])
    M_j = M[candidates]

    t_start_i = np.argmax(M[i] != 0)
    t_end_i = np.argmax(np.cumsum(M[i]))

    # Start and end times for candidates.
    t_start_j = np.argmax(M_j != 0, axis=1)
    t_end_j = np.argmax(np.cumsum(M_j, axis=1), axis=1)

    eligeble = np.logical_and(t_start_j <= t_start_i, t_end_i <= t_end_j)

    # Candidates starting no later than m_i and ending no sooner than m_i.
    return M_j[eligeble]


def align(M_j, taus):
    
    M_j_aligned = []
    for m_j, tau in zip(M_j, taus):
        M_j_aligned.append(shift(m_j, -tau, fill_value=0))
        
    return np.array(M_j_aligned)


def synthesize(i, m_i, N_i):
        
    model = LinearRegression(fit_intercept=False)
    model.fit(np.transpose(N_i), m_i)
    
    return np.transpose(N_i) @ model.coef_
    
    #beta = np.linalg.inv(N_i @ N_i.T) @ N_i @ m_i
    
    #return N_i.T @ beta


if __name__ == "__main__":

    from utils import interpolation_region_mask

    Y = np.load("../../data/graph_learning/Y.npy")
    Z = np.load("../../data/graph_learning/Z.npy")
    O = interpolation_region_mask(Y)
    M = Z * O 
    
    np.random.seed(42)    
    for i in range(12):

        N_i = np.asarray(np.random.choice(range(M.shape[0]), size=10, replace=False))
    
        # Find optimal alignment of elegible neighbours.
        tau_i_star, N_i_star, C_i_star = cross_correlation(i=i, N_i=N_i, M=M, eps=0.5)
    
        aligned_neighbours = align(M[N_i_star], tau_i_star)

