import numpy as np 

from scipy.sparse import csgraph

from profile_synthesis import cross_correlation, synthesize, align
from aquisition import candidate_adjacency


def candidate_schedule(T0, x):

    return np.ceil(0.5 * T0 * (1 - np.tanh(10 * x / len(x) - 5)) + 1).astype(int)


def annealing_schedule(T0, x):
    
    return 0.5 * T0 * (1 - np.tanh(10 * x / len(x) - 5))


def energy(M, M_hat, A, alpha):
    """Evaluate \min_{A} | M - M^* |_F^2 + \gamma tr((M^*)^\top L M^*)."""
    
    L = csgraph.laplacian(A, normed=True)
    
    loss_local = np.linalg.norm(M - M_hat) ** 2
    loss_global = np.trace(M_hat.T @ L @ M_hat)
    
    return alpha * loss_local + (1 - alpha) * loss_global


def p_accept(E, E_new, T):
    
    if E_new < E:
        return 1.0
    
    return np.exp(-1.0 * (E_new - E) / T)


def run_step(A, M, to_update=None, alpha=0.5, eps=0):
    
    # Refined adjacency after checking correlations.
    A_refined = np.zeros_like(A)

    # Update all nodes by default.
    if to_update is None:
        to_update = np.arange(A.shape[0])
    
    M_hat = np.zeros_like(M)
    for i in to_update:
        
        a = A[i]
        
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
        
        M_hat[i] = synthesize(i, M[i], align(M[N_i_star], tau_i_star))
        
    # Just in case.
    np.fill_diagonal(A_refined, 0)

    # Sanity check.
    assert np.array_equal(A_refined, A_refined.T)
    
    return A_refined, energy(M, M_hat, A_refined, alpha), M_hat


def estimate_adjacency(A, M, num_epochs, N,
                       eps=0.5, alpha=0.5, seed=42, patience=10, epsilon=1e-6):
    """
    Args:
        patience: Maximum number of iterations without an update before exiting.
    """
    
    # Initial adjacency and energy.
    A, E, M_hat = run_step(A=A, M=M, alpha=alpha, eps=eps)
    
    rnd_state = np.random.RandomState(seed=seed)

    # Temperature should be at the magnitude of energy levels.
    T = annealing_schedule(T0=E, x=np.arange(num_epochs))
    S = candidate_schedule(T0=N, x=np.arange(num_epochs))

    loss = np.linalg.norm(M - M_hat, axis=1) ** 2
    for epoch, T_n in enumerate(T):

        A_new = candidate_adjacency(A, loss, n_active=S[epoch], 
                                    update_j=False, rnd_state=rnd_state)
        
        A_new, E_new, M_hat_new = run_step(A_new, M=M, to_update=np.squeeze(np.where(loss > epsilon)),
                                           eps=eps, alpha=alpha)

        # Move on to next state.
        if p_accept(E, E_new, T_n) >= np.random.random():
                        
            A = A_new
            E = E_new
            M_hat = M_hat_new
            
            loss = np.linalg.norm(M - M_hat, axis=1) ** 2

        else:

            patience -= 1    
            if patience < 1:

                print(f"!!!Early stopping after {epoch + 1} iterations!!!")
                return A
    
    return A