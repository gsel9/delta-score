"""
"""

from typing import Union, List

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import betabinom

# TEMP: Add . before imports!
from transition import next_state, inital_state
from sojourn import sojourn_time 


# TODO: 
# * Add config module with constants.
# * Extend time grid to arbitrary number of datapoints.
def simulate_profile(n_timepoints: int, missing=0) -> np.ndarray:
    """Update the profile vector of a single female. 

    Returns:
        Simulated screening history for one single female.
    """

    age = 16 # randomly sample
    age_max = 96 # randomly sample betabinom(1, )

    t_grid = np.linspace(age, age_max, n_timepoints)

    x = np.ones(n_timepoints) * missing
    
    state = inital_state(init_age=age)

    # NOTE: 
    # * age, t_exit are defined on grid [16, 96].
    # * a, b are defined on grid [0, n_timepoints].

    i, a, b = 0, 0, 0
    while age < age_max:

        # Exit time from current state.
        t_exit = int(sojourn_time(age, age_max, state))

        #b = np.argmax(np.cumsum(t_exit + age <= t_grid))
        b = t_exit + age

        # Number of timepoints to cover in state vector.
        #if t_exit >= age_max:
        #    b = age_max

        #else:
        #    b = np.argmax(np.cumsum(t_exit - age < t_grid))

        x[a:b] = state

        # Update age.
        age = age + t_exit

        # Update state.
        state = next_state(age, state)

        # To avoid endless loop.
        i += 1
        if i > age_max:
            raise RuntimeError('Endless loop. Check config!')

        # Update auxillary variable.
        a = b 

    return x


if __name__ == "__main__":
    # TEMP: Development

    # TARGET:
    # [1. 2. 3. 4.] [14899   815   188    17] [0.93592562 0.05119668 0.01180979 0.00106791]

    # TODO: 
    # * Sample HMM censoring times from a beta-binomial with alpha = 4.57; beta = 5.47
    # * Sample time for first screening analytically by fitting a distribution to empirical data.
    # * Update inital state probas and transit intensities. 

    n_timepoints = 100

    # Number of screening histories/females/samples.
    n_samples = 200

    np.random.seed(42)

    D = []
    for num in range(n_samples):

        # Simulate a synth screening profile.
        d = simulate_profile(n_timepoints)
        
        if sum(d) == 0:
            continue

        D.append(d)

    D = np.array(D)
    
    v, c = np.unique(D[D != 0], return_counts=True)
    print(v, c, c / sum(c))

    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.imshow(D, aspect="auto")
    # plt.show()

    # idx = np.squeeze(np.where(np.max(D, axis=1) > 1))
    # _, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 15))
    # for i, axis in enumerate(axes.ravel()):

    #     x = D[idx[i], :]
    #     x[x == 0] = np.nan
    #     axis.plot(x, "o")
    # plt.show()
