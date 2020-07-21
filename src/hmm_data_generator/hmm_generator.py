"""
"""

from typing import Union, List

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

# TEMP: Add . before imports!
from transition import next_state, inital_state
from sojourn import sojourn_time 


# TODO: Add config module with constants.
def simulate_profile(n_timepoints, missing=0) -> np.ndarray:
    """Update the profile vector of a single female. 

    Returns:
        Simulated screening history for one single female.
    """

    # TEMP: init_age and age_max from analytical distributions. 
    init_age = 16
    age_max = 96

    x = np.ones(int(n_timepoints)) * missing

    # Initial state.
    current_state = inital_state(init_age=init_age)

    # TODO: How time grid should be defined.
    taus = np.linspace(init_age, age_max, n_timepoints)
    taus = np.insert(taus, 0, 0)
    ###
    
    # Track age development.
    current_age = init_age

    # Counters. 
    t_start = init_age
    t_end = 0

    _iter = 0
    while current_age < age_max:

        # When female leaves the current state.
        t_exit = sojourn_time(current_age, age_max, current_state)

        current_age = current_age + t_exit
        current_state = next_state(current_age, current_state)

        t_start = 
        t_end = 

        x[t_start:t_end] = current_state

        # To avoid endless loop.
        _iter += 1
        if _iter > n_timepoints:
            raise RuntimeError('Endless loop. Check config!')

    return x


if __name__ == "__main__":
    # TEMP: Development

    # TODO: 
    # * Sample HMM censoring times from a beta-binomial with alpha = 4.57; beta = 5.47
    # * Sample time for first screening analytically by fitting a distribution to empirical data.
    # * Update inital state probas and transit intensities. 

    n_timepoints = 340

    # Number of screening histories/females/samples.
    n_samples = 2

    np.random.seed(42)

    D = []
    for num in range(n_samples):

        # Simulate a synth screening profile.
        d = simulate_profile(n_timepoints)

        #if sum(d) == 0:
        #    continue

        #D.append(d)

    #print(D)