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
    init_age = 0
    age_max = 80
    current_age = init_age

    x = np.ones(int(n_timepoints)) * missing

    # Initial state.
    current_state = inital_state(init_age=init_age)

    _iter = 0
    while current_age < age_max:

        t_exit = int(sojourn_time(current_age, age_max, current_state))

        t_end = current_age + t_exit

        x[current_age:t_end] = current_state

        current_state = next_state(current_age, current_state)

        current_age = t_end

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

    n_timepoints = 80

    # Number of screening histories/females/samples.
    n_samples = 100

    np.random.seed(42)

    D = []
    for num in range(n_samples):

        # Simulate a synth screening profile.
        d = simulate_profile(n_timepoints)
        
        if sum(d) == 0:
            continue

        D.append(d)

    D = np.array(D)
    print(D)

    v, c = np.unique(D[D != 0], return_counts=True)
    print(v, c, c / sum(c))
