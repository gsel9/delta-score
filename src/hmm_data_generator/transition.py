"""
"""

from typing import List, Union

import numpy as np

# TEMP: Add . before imports!
from utils import age_group_idx, lambda_sr, p_init_state


def inital_state(init_age: int) -> int:
    """Sample the state at first screening."""
    
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


def next_state(age: int, current_state: int, censoring: int = 0) -> int:
    """Simulate the next state from sojourn time conditions.

    Args:
        age:
        current_state: 
        censoring: Representation of censoring.

    Returns:
        The next state.
    """

    p = legal_transitions(current_state, lambda_sr[age_group_idx(age)], norm=True)

    # s1 -> s2 or s1 -> censoring.
    if current_state == 1:
        return np.random.choice((2, censoring), p=p)

    # s2 -> s3 or s2 -> s1 or -> censoring.
    if current_state == 2:
        return np.random.choice((3, 1, censoring), p=p)
    
    # s3 -> s4 or s3 -> s2 or -> censoring.
    if current_state == 3:
        return np.random.choice((4, 2, censoring), p=p)
    
    # s4 -> s1 or s4 -> censoring.
    if current_state == 4:
        return np.random.choice((1, censoring), p=p)

    return censoring


if __name__ == '__main__':
    next_state(16, 2)
