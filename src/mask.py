"""
Algorithms for simulating matrices that exhibit the same characteristics as Norwegian cervical cancer screening data.
"""

import numpy as np


def sample_start_age(n_timepoints, proba_init_age, return_idx=True):
    """
    Returns:
        Age at inital screening.
    """
    
    # Randomly select location of initial age along a defined time axis.
    time = np.linspace(0, n_timepoints - 1, n_timepoints)
    
    init_age_idx = np.random.choice(range(n_timepoints), p=proba_init_age)
    init_age = int(time[init_age_idx])
    
    if return_idx:
        return init_age, init_age_idx
    
    return init_age


def sample_end_age(n_timepoints, proba_dropout, init_age_idx=0):
    """
    Returns:
        Age at final screening.
    """
    time = np.linspace(0, n_timepoints - 1, n_timepoints)
    
    p = proba_dropout[init_age_idx:] / sum(proba_dropout[init_age_idx:])
    
    return int(np.random.choice(time[init_age_idx:], p=p))


def sample_screenings(X, stepsize, proba_init_age=None, proba_dropout=None, missing=0):

    n_timepoints = X.shape[1]

    X_sparse = []
    for num, x in enumerate(X): 

        if proba_init_age is None:
            min_age = np.argmax(x)

        else:
            min_age, min_age_idx = sample_start_age(n_timepoints, proba_init_age)

        if proba_dropout is None:
            max_age = np.argmax(np.cumsum(x))

        else:
            max_age = sample_end_age(n_timepoints, proba_dropout, min_age_idx)

        # Sanity check.
        assert min_age < max_age + 1

        to_keep = np.arange(min_age, max_age, 1)[::stepsize]

        x_sparse = np.zeros_like(x)
        x_sparse[to_keep] = x[to_keep]
        
        if sum(x[to_keep]) == 0:
            continue

        X_sparse.append(x_sparse)

    return np.array(X_sparse)


def simulate_mask(X_integer,
                  mask_parameters,
                  path_dropout=None,
                  seed=42):
    """Simulation of a missing data mask.
    Parameters
    ----------
    X_integer : The unmasked integer-valued matrix.
    mask_parameters : 
        mask_transition_expectations : E[p_ik] for k = 1, 2, ... Z
        mask_transition_variances : Var[p_ik] for k = 1, 2, ... Z
        memory_length : Determines how long each screening result is remembered.
        mask_level : Affects the probability of coming in for a screening across all patients.
    Returns
    ----------
    mask : The resulting mask.
    """
    mask_screening_proba = mask_parameters['mask_screening_proba']
    mask_memory_length = mask_parameters['memory_length']
    mask_level = mask_parameters['mask_level']

    N, T = np.shape(X_integer)

    mask = np.zeros_like(X_integer, dtype=np.bool)
    observed_values = np.zeros_like(X_integer)

    np.random.seed(seed)

    for t in range(T - 1):

        # Find last remembered values.
        last_remembered_values = observed_values[np.arange(
            N), t + 1 - np.argmax(observed_values[:, t + 1:max(0, t - mask_memory_length):-1] != 0, axis=1)]

        p = mask_level * mask_screening_proba[(last_remembered_values).astype(int)]
        r = np.random.uniform(size=N)
        mask[r <= p, t + 1] = True
        observed_values[r <= p, t+1] = X_integer[r <= p, t+1]

    # Simulate dropout
    if path_dropout is not None:
        prob_dropout = np.load(path_dropout)
        tpoints = np.arange(X_integer.shape[1])

        for num in range(X_integer.shape[0]):
            t_max = np.random.choice(tpoints, p=prob_dropout, replace=True)
            mask[num, t_max:] = 0

    return mask