from scipy import signal
from scipy.sparse import csgraph
from sklearn.linear_model import Ridge

from sklearn.metrics import pairwise_distances, jaccard_score

import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt


# TODO: Numba decorator!!!


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


def energy(M, M_hat, A, beta):
    
    L = csgraph.laplacian(A, normed=True)
    
    loss_local = np.linalg.norm(M - M_hat) ** 2
    loss_global = np.trace(M_hat.T @ L @ M_hat)
    
    return beta * loss_local + (1 - beta) * loss_global


def neighbourhood_similarity(N_i, N_j):
    
    return jaccard_score(N_i != 0, N_j != 0)


def annealing_schedule(T0, x):
    
    return 0.5 * T0 * (1 - np.tanh(10 * x / len(x) - 5))


def p_accept(E, E_new, T):
    
    if E_new < E:
        return 1.0
    
    return np.exp(-1.0 * (E_new - E) / T)


def cross_correlation(i, N_i, M, eps=0.5):
    """
    Args:
        i: Index of node i.
        N_i: Indicies for the neighbours of i.
        M: Graph signals.
        eps: Profile correlation treshold.
    """
    
    # NB: Need N_i to contain the actual neighbour indicies.
    N_i = check_iterable(N_i)
    
    tau_i_star, N_i_star, C_i_star = [], [], []
    for k, m_j in enumerate(M[N_i]):
    
        c = signal.correlate(m_j, M[i])
        c_max = max(c) / sum(c != 0)

        # Retain only the neighbours sufficiently correlated.
        if c_max >= eps:
                   
            tau_i_star.append(np.argmax(c) + 1 - len(m_j))
            
            N_i_star.append(N_i[k])
            C_i_star.append(c_max)

    return tau_i_star, N_i_star, C_i_star


def align(M_j, taus):
    
    M_j_aligned = []
    for m_j, tau in zip(M_j, taus):
        
        T_j = np.argmax(m_j != 0)
        
        if tau > 0:
            k = T_j - tau
            
        elif tau < 0:
            k = abs(tau) + T_j

        else:
            k = T_j
        
        m_j_shifted = np.zeros_like(m_j)
        
        if k < 0:

            sliced_m_j = m_j[m_j > 0][abs(k):]   
            m_j_shifted[:len(sliced_m_j)] = sliced_m_j

        else:
            m_j_shifted[k:sum(m_j > 0) + k] = m_j[m_j > 0]
        
        M_j_aligned.append(m_j_shifted)
        
    return np.array(M_j_aligned)


def synthesize(i, m_i, N_i, alpha=0):
        
    #model = Ridge(alpha=alpha, fit_intercept=False)
    #model.fit(np.transpose(N_i), m_i)
    
    #return np.transpose(N_i) @ model.coef_
    
    beta = np.linalg.inv(N_i @ N_i.T) @ N_i @ m_i
    
    return N_i.T @ beta


def candidate_adjacency(A, loss, n_active, n_candidates, rnd_state, update_j=False):
    """
    Args:
        A: Current adjacency matrix.
        n_active: Number of neighbourhoods to update.
        n_candidates: Number of neighbourhoods to compare in an update.
        update_j: Update neighbourhoods of both i and j.
    """
    
    active = rnd_state.choice(range(A.shape[0]), replace=False, 
                              size=min(A.shape[0], n_active), p=loss / sum(loss))
    A_new = A.copy()    

    for i in active:

        # Node is not connected.
        if sum(A[i]) < 1:
            A_new = random_update(A, A_new, i, rnd_state)
            
        else:
            A_new = cf_update(A, A_new, i, n_candidates, rnd_state)

    # Just in case.
    np.fill_diagonal(A_new, 0)

    # Sanity check.
    assert np.array_equal(A_new, A_new.T)
                
    return A_new


def cf_update(A, A_new, i, n_candidates, rnd_state):
    
    # Ensure i not in N_j.
    candidates = np.concatenate([np.arange(i), np.arange(i + 1, A.shape[0])])
    candidates = rnd_state.choice(candidates, replace=False, size=n_candidates)
    
    new_hood = None
    max_overlap = 0

    for j in candidates:

        overlap = neighbourhood_similarity(A[i], A[j])
        if overlap > max_overlap:

            new_hood = j
            max_overlap = overlap

    return update_neighbourhood(max_overlap, A_new, A, i, j, update_j=False)


def random_update(A, A_new, i, rnd_state):
    
    candidates = np.concatenate([np.arange(i), np.arange(i + 1, A.shape[0])])
    candidates = np.asarray([rnd_state.choice(candidates)])
        
    return update_neighbourhood(overlap=0, A_new=A_new, A=A, i=i, j=candidates, update_j=False)


def update_neighbourhood(overlap, A_new, A, i, j, update_j=False):

    # Connect i and j.
    if overlap == 1:

        A_new[i, j] = 1
        A_new[j, i] = 1

    else:

        # Union of both neighbourhoods.
        N_new = np.squeeze(np.logical_or(A[i], A[j]))

        A_new[i, N_new] = 1
        A_new[N_new, i] = 1
        
        if update_j:
        
            A_new[j, N_new] = 1
            A_new[N_new, j] = 1

    return A_new


def run_step(A, M, alpha=0, beta=0, eps=0):
    
    # Refined adjacency after checking correlations.
    A_refined = np.zeros_like(A)
    
    M_hat = np.zeros_like(M)
    for i, a in enumerate(A):
        
        # Node is not connected.
        if sum(a) < 1:
        
            M_hat[i] = np.zeros_like(M[i])
            continue
        
        tau_i_star, N_i_star, C_i_star = cross_correlation(i=i, N_i=np.squeeze(np.where(a)), M=M, eps=eps)
        
        # Avoid sum check as node can be connected to zero-index node.
        if np.size(N_i_star) == 0:
            
            M_hat[i] = np.zeros_like(M[i])
            continue
        
        A_refined[i, N_i_star] = 1
        A_refined[N_i_star, i] = 1
        
        M_hat[i] = synthesize(i, M[i], align(M[N_i_star], tau_i_star), alpha=alpha)
        
    # Just in case.
    np.fill_diagonal(A_refined, 0)

    # Sanity check.
    assert np.array_equal(A_refined, A_refined.T)
    
    return A_refined, energy(M, M_hat, A_refined, beta), M_hat


def main_procedure(A, M, num_epochs, n_active, n_candidates, 
                   eps=0.5, alpha=0, beta=0, seed=42, patience=10):
    """
    Args:
        patience: Maximum number of iterations without an update before exiting.
    """
    
    # Initial adjacency and energy.
    A, E, M_hat = run_step(A=A, M=M, alpha=alpha, beta=beta, eps=eps)
    
    # Temperature should be at the magnitude of energy levels.
    T = annealing_schedule(T0=E, x=np.arange(num_epochs))
    
    # Initialize loss.
    loss = np.linalg.norm(M - M_hat, axis=1) ** 2
    
    rnd_state = np.random.RandomState(seed=seed)
    
    for epoch, T_n in enumerate(T):
        
        A_new = candidate_adjacency(A, loss, n_active, n_candidates, update_j=False, rnd_state=rnd_state)
        
        A_new, E_new, M_hat_new = run_step(A_new, M=M, eps=eps, alpha=alpha, beta=beta)

        # Move on to next state.
        if p_accept(E, E_new, T_n) >= np.random.random():
                        
            A = A_new
            E = E_new
            M_hat = M_hat_new
            
            loss = np.linalg.norm(M - M_hat, axis=1) ** 2
        
        # TODO: Break if not updated in the last `patience` number of iterations.
        if epoch > patience:
            return A
    
    return A



if __name__ == "__main__":


	def interpolation_region_mask(Y):
    
	    O = np.zeros_like(Y)
	    for i, y in enumerate(Y):
	        
	        t_start = np.argmax(Y[i] != 0)
	        t_end = np.argmax(np.cumsum(Y[i])) + 1
	        
	        O[i, t_start:t_end] = 1
	        
	    return O


	def synthetic_data_gen():
    
	    M = np.load("../data/M_train.npy")
	    Y = np.load("../data/X_train.npy")
	    
	    O = interpolation_region_mask(Y)

	    return M * O
	    
	    
	def screening_data_gen():
	    
	    M = np.load("/Users/sela/Desktop/recsys_paper/results/screening/mf/train/train_Xrec.npy")
	    Y = np.load("/Users/sela/Desktop/recsys_paper/data/screening/train/X_train.npy")
	    O = interpolation_region_mask(Y)

	    return M * O


	# Choose between synthetic and screening data for demonstration run.
	#M = synthetic_data_gen()
	M = screening_data_gen()

	# Examples of profiles.
	#_, axes = plt.subplots(ncols=3, nrows=4, figsize=(12, 8))
	for i, axis in enumerate(axes.ravel()):
	    axis.plot(M[i])
	plt.tight_layout()

	n = 6
	A_init = intial_adjacency(n=n, p=0.4, seed=42)
	print(A_init)
	print()

	# Increase n_active to increase density in A. 
	# Increase eps to decrease density in A. 
	A_hat = main_procedure(A_init, M[:n], num_epochs=10, eps=0.55, n_active=2, n_candidates=4)
	#np.save("A_hat.npy", A_hat)
	print(A_hat)

	np.array_equal(A_init, A_hat), np.sum(A_init), np.sum(A_hat)