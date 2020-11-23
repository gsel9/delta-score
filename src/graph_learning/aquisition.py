import numpy as np 

from sklearn.metrics import jaccard_score


# See also: https://stackoverflow.com/questions/49733244/how-can-i-calculate-neighborhood-overlap-in-weighted-network
def neighbourhood_similarity(N_i, N_j):
    
    return jaccard_score(N_i != 0, N_j != 0)


def candidate_adjacency(A, loss, n_active, rnd_state, update_j=False):
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
            A_new = cf_update(A, A_new, i, n_active, rnd_state)

    # Just in case.
    np.fill_diagonal(A_new, 0)

    # Sanity check.
    assert np.array_equal(A_new, A_new.T)
                
    return A_new


def cf_update(A, A_new, i, n_active, rnd_state):
    
    # Ensure i not in N_j.
    candidates = np.concatenate([np.arange(i), np.arange(i + 1, A.shape[0])])
    candidates = rnd_state.choice(candidates, replace=False, size=n_active)
    
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
