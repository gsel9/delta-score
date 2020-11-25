import numpy as np 

from scipy.sparse import csgraph

try:
    from .profile_synthesis import cross_correlation, synthesize, align
    from .aquisition import candidate_adjacency
except:
    from profile_synthesis import cross_correlation, synthesize, align
    from aquisition import candidate_adjacency


def candidate_schedule(T0, x):

    return np.ceil(0.5 * T0 * (1 - np.tanh(10 * x / len(x) - 5)) + 1).astype(int)


def annealing_schedule(T0, x):
    
    return 0.5 * T0 * (1 - np.tanh(10 * x / len(x) - 5))


def energy(M, M_hat, A, alpha):
    """Evaluate \min_{A} | M - M^* |_F^2 + \gamma tr((M^*)^\top L M^*)."""
    
    loss_local = np.linalg.norm(M - M_hat) ** 2

    L = csgraph.laplacian(A, normed=True)
    loss_global = np.trace(M_hat.T @ L @ M_hat)
    
    return alpha * loss_local + (1 - alpha) * loss_global


def p_accept_adjacency(E, E_new, T):
    
    if E_new < E:
        return 1.0
    
    return np.exp(-1.0 * (E_new - E) / T)


# OPTIMIZE: Recompute approximation only at the updated nodes.
def evaluate_adjacency(A, M, to_update=None, alpha=0.5):
    
    # Update all nodes by default.
    if to_update is None:
        to_update = np.arange(A.shape[0])
    
    M_hat, C_star = np.zeros_like(M), []
    for i in to_update:

        a = A[i]
        
        # Node is not connected.
        if sum(a) < 1:
        
            M_hat[i] = np.zeros_like(M[i])
            C_star.append([])
            continue
        
        tau_i_star, N_i_star, C_i_star = cross_correlation(i=i, N_i=np.squeeze(np.where(a)), M=M)

        # Avoid sum check as node can be connected to zero-index node.
        if np.size(N_i_star) < 1:
            
            M_hat[i] = np.zeros_like(M[i])
            C_star.append([])

            continue
        
        M_hat[i] = synthesize(i, M[i], align(M[N_i_star], tau_i_star))

        C_star.append(np.array(C_i_star))
        
    return energy(M, M_hat, A, alpha), C_star, M_hat


def estimate_adjacency(A, M, num_epochs, n, k, method,
                       alpha=0.5, seed=42, patience=10, epsilon=1e-6):
    """
    Args:
        n: The number of neighbourshoods to update in each iterations.
        k: The number of neighbours to sample for each node.
        patience: Maximum number of iterations without an update before exiting.
    """
    
    # Initialize variables.
    E, C, M_hat = evaluate_adjacency(A=A, M=M, alpha=alpha)
    
    rnd_state = np.random.RandomState(seed=seed)

    # Temperature should be at the magnitude of energy levels.
    T = annealing_schedule(T0=E, x=np.arange(num_epochs))
    #S = candidate_schedule(T0=n_candidates, x=np.arange(num_epochs))

    loss = np.linalg.norm(M - M_hat, axis=1) ** 2

    energies, losses = [E], [np.linalg.norm(M - M_hat)]
    for epoch, T_n in enumerate(T):
        
        A_new = candidate_adjacency(method, A, loss, C=C, n=n, k=k, rnd_state=rnd_state)

        E_new, C_new, M_hat_new = evaluate_adjacency(A_new, M=M, alpha=alpha)

        # Move on to next state.
        if p_accept_adjacency(E, E_new, T_n) >= rnd_state.random():

            #print(f"Accepting at E_new={E_new} (E previous={E})")
                        
            A = A_new
            C = C_new
            E = E_new
            M_hat = M_hat_new
            
            loss = np.linalg.norm(M - M_hat, axis=1) ** 2

            losses.append(np.linalg.norm(M - M_hat))
            energies.append(E)

        else:

            patience -= 1    
            if patience < 1:

                print(f"!!!Early stopping after {epoch + 1} iterations!!!")
                return A, M_hat, energies, losses
    
    return A, M_hat, energies, losses
