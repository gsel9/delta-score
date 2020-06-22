"""
"""

from typing import Union, List

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from .transition import next_state, inital_state
from .sojourn import sojourn_time 


# TODO: Randomly sample init_age and age_max with empirical probabilities.
def simulate_profile(n_timepoints, init_age, age_max, missing=0) -> np.ndarray:
    """Update the profile vector of a single female. 

    Returns:
        Simulated screening history for one single female.
    """

    x = np.ones(int(n_timepoints)) * missing

    # Initial state.
    current_state = inital_state(init_age=init_age)
    
    # Track age development.
    current_age = init_age

    # Counters. 
    start_period = init_age
    end_period = 0

    _iter = 0
    while current_age < age_max:

        # Time spent in current state.
        dt = sojourn_time(current_age, age_max, current_state)

        end_period = end_period + int(dt)
        current_age = current_age + int(dt)

        x[start_period:end_period] = current_state

        start_period = end_period
        prev_state = current_state

        # Update profile values with current state.
        current_state = next_state(age=current_age, current_state=current_state, censoring=0)
        print("AGE", current_age)
        # To avoid endless loop.
        _iter += 1
        if _iter > n_timepoints:
            raise RuntimeError('Endless loop. Check config!')

    return x
