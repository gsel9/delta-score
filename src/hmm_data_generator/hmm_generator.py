from tqdm import tqdm

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

# TEMP: Add . before imports!
from transition import next_state, inital_state
from sojourn import time_exit_state


# TEMP: Should compare discret distributions.
def times_first_screen_censoring(n_samples, n_timepoints):
    """Sample times for init and final screenings."""

    print(f"Sampling {n_samples} time points.")

    t_start = st.exponnorm.rvs(K=8.76, loc=9.80, scale=7.07, size=n_samples)
    t_start = t_start.astype(int)

    t_cens = st.exponweib.rvs(a=513.28, c=4.02, loc=-992.87, scale=707.63, size=n_samples)
    t_cens = t_cens.astype(int)

    # Sanity check
    assert len(t_start) == len(t_cens)

    # NB: Clip inital screening times to temporal grid.
    t_start[t_start < 0] = 0

    # NB: Clip censoring times to temporal grid.
    t_cens[t_cens > n_timepoints] = n_timepoints

    # NB: Make sure t_end > t_start for all females.
    i = t_start < t_cens
    t_cens = t_cens[i]
    t_start = t_start[i]

    # Sanity check
    assert np.all(t_start < t_cens)

    print(f"Kept {sum(i)} time points.")

    return t_start, t_cens


def simulate_profile(age, age_max, n_timepoints, censoring=0) -> np.ndarray:
    """Returns a simulated screening history for one single female.
    """

    x = np.ones(n_timepoints) * censoring
    
    state = inital_state(init_age=age)

    period_start = age
    while age < age_max:

        # Age at exit time from current state.
        age = int(round(time_exit_state(age, age_max, state)))

        # Update state vector.
        x[period_start:age] = state

        state = next_state(age, state)

        # Censoring rest of state vector.
        if state == censoring:
            return x

        # Update auxillary variable.
        period_start = age

    return x


# NB: Expect n_timepoints = 321 in utils.py.
def simulate_screening_histories(n_samples, n_timepoints=321):

    t_start, t_cens = times_first_screen_censoring(n_samples, n_timepoints=n_timepoints)

    D = []
    for t_a, t_b in tqdm(zip(t_start, t_cens), total=len(t_start)):

        d = simulate_profile(t_a, t_b, n_timepoints)

        # NOTE: Require at least 5 obs.
        if sum(d != 0) < 5:
            continue

        D.append(d)

    return np.array(D)


if __name__ == "__main__":
    # Demo run.
    np.random.seed(42)

    D = simulate_screening_histories(n_samples=50000)

    # Analyse histories.
    v, c = np.unique(D[D != 0], return_counts=True)
    print(v, c, c / sum(c))

    np.save("/Users/sela/Desktop/hmm.npy", D)
    D = np.load("/Users/sela/Desktop/hmm.npy")

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
    #     #axis.set_xlim(0, 321)

    #     #y = np.ones(321) * np.nan
    #     #y[x != 0] = x[x != 0]
    #     #axis.plot(y, "o")

    # plt.show()
 