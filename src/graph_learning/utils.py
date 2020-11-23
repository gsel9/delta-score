import numpy as np 
import networkx as nx

from sklearn.metrics import jaccard_score


def intial_adjacency(n, p, seed):
    
    # Initialize with random adjancey.
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=False)
    return nx.to_numpy_array(G)


def check_iterable(N_i):
    """Check the data structure of a neighbourhood."""
    
    if isinstance(N_i, (int, float)):
        N_i = np.array([int(N_i)])
        
    if not isinstance(N_i, np.ndarray):
        N_i = np.array(N_i)
        
    if np.ndim(N_i) < 1:
        N_i = np.expand_dims(N_i, axis=0)
        
    return N_i