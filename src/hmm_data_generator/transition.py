"""
"""

from typing import List, Union

import numpy as np

from utils import age_group_idx, lambda_sr, p_init_state


def inital_state(init_age: int) -> int:
    """Sample a state at first screening."""
    
    return np.random.choice([1, 2, 3, 4], p=p_init_state[age_group_idx(init_age)])       


# QUESTION: How is treatment given by transition intensities???
def legal_transitions(current_state: int, age_group_idx: int) -> np.ndarray:
    """Filter intensities for shifts from the current state."""

    # Transition intensities for the given age group.
    lambdas = lambda_sr[age_group_idx]
    
    # Censoring.
    if current_state == 0:
        return [0]

    # N0 -> L1/D4.
    if current_state == 1:
        return [lambdas[0], lambdas[5]]
    
    # L1 -> N0/H2/D4.
    if current_state == 2:
        return [lambdas[3], lambdas[1], lambdas[6]]

    # H2 -> N0/C3/D4
    if current_state == 3:
        return [lambdas[4], lambdas[2], lambdas[7]]

    # C3 -> D4
    if current_state == 4:
        return [lambdas[8]]

    # Normalise into proabilities.
    #return np.array(l_sr) / sum(l_sr)


def next_state(age_exit: int, current_state: int, censoring: int = 0) -> int:
    """Returns next female state."""

    lambdas = legal_transitions(current_state, age_group_idx(age_exit))

    if len(lambdas) == 1:
        return censoring

    p = np.array(lambdas) / sum(lambdas)

    # N0 -> L1/D4 (censoring)
    if current_state == 1:
        return np.random.choice((2, censoring), p=p)

    # L1 -> N0/H2/D4
    if current_state == 2:
        return np.random.choice((1, 3, censoring), p=p)
    
    # H2 -> N0/C3/D4
    if current_state == 3:
        return np.random.choice((2, 4, censoring), p=p)
    
    # C3 -> D4
    return censoring


if __name__ == '__main__':
    print(next_state(16, 2))
