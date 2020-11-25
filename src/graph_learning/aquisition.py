import numpy as np 

from sklearn.metrics import jaccard_score

try:
    from .utils import check_adjacency, check_iterable
except:
    from utils import check_adjacency, check_iterable


# See also: https://stackoverflow.com/questions/49733244/how-can-i-calculate-neighborhood-overlap-in-weighted-network
def neighbourhood_similarity(N_i, N_j):
    
    return jaccard_score(N_i != 0, N_j != 0)


def random_candidate_adjacency(A, loss, n, k, rnd_state):

    # Prioretize nodes with largest loss.
    active = rnd_state.choice(np.arange(A.shape[0]), replace=False, 
                              size=min(A.shape[0], n), p=loss / sum(loss))

    # Record the nodes that have changes to their neighbourhoods.
    updated = []

    A_new = A.copy()
    for i in active:

        # Kill signal from previous neighbours.
        A_new[i, :] = 0
        A_new[:, i] = 0

        A_new = random_neighbours(A_new, i, k, rnd_state)
            
    return check_adjacency(A_new)

def random_neighbours(A_new, i, k, rnd_state):
    """Assign `k` random neighbours to node i."""

    candidates = np.concatenate([np.arange(i), np.arange(i + 1, A_new.shape[0])])
    N_new = rnd_state.choice(candidates, replace=False, size=k)
    
    A_new[i, N_new] = 1
    A_new[N_new, i] = 1

    return A_new


def candidate_adjacency(method, A, loss, C, n, k, rnd_state, update_j=False):

    if method == "random":
        return random_candidate_adjacency(A, loss, n, k, rnd_state)

    if method == "cf":
        return cf_candidate_adjacency(A, loss, n, k, C, rnd_state)

    raise ValueError(f"Invalid method {method}")


def cf_candidate_adjacency(A, loss, n, k, C, rnd_state):
    
    # Prioretize nodes with largest loss.
    active = rnd_state.choice(np.arange(A.shape[0]), replace=False, 
                              size=min(A.shape[0], k), p=loss / sum(loss))

    A_new = A.copy()
    for i in active:

        # Node is not connected to any neighbours.
        if sum(A[i]) < 1:
            A_new = random_neighbours(A_new, i, k, rnd_state)
            
        else:
            A_new = cf_neighbours(A, A_new, C, i, k, rnd_state)
                        
    return check_adjacency(A_new)


def cf_neighbours(A, A_new, C, i, k, rnd_state):

    # Sample neighbours to compare with. Ensure i not in N_j.
    # Do random sampling to include some new candidates in each round.
    candidates = np.concatenate([np.arange(i), np.arange(i + 1, A.shape[0])])
    candidates = rnd_state.choice(candidates, replace=False, size=k)

    compare_to_i = lambda A_j: neighbourhood_similarity(A[i], A_j)
    overlaps = np.apply_along_axis(compare_to_i, 1, A[candidates])
    
    new_hood = candidates[np.argmax(overlaps)]
    max_overlap = max(overlaps)

    if np.isclose(max_overlap, 0):

        return random_neighbours(A_new, i, k, rnd_state)

    if np.isclose(max_overlap, 1):

        A_new[i, new_hood] = 1
        A_new[new_hood, i] = 1

        return A_new

    return cf_update_neighbourhood(max_overlap, A_new, A, i, new_hood, C, rnd_state)


def cf_update_neighbourhood(overlap, A_new, A, i, j, C, rnd_state):
    # Replacing the least correlated neighs of i with the new neighs of j.

    N_i = check_iterable(np.squeeze(np.where(A[i])))

    # Find the neighbours of j that are not common to i.
    candidates = np.setdiff1d(np.where(A[j]), N_i)

    # Rank neighbours by correlation to i.
    to_replace = N_i[np.argsort(C[i])][:len(candidates)]

    # Replace the least correlated neighbours of i.
    A_new[to_replace, i] = 0
    A_new[i, to_replace] = 0

    A_new[candidates, i] = 1
    A_new[i, candidates] = 1

    # Force diagonal since i could be in `candidates`.
    np.fill_diagonal(A_new, 0)

    return A_new
