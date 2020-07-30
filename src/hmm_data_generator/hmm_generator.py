import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

from transition import next_state, inital_state
from sojourn import time_exit_state


# TEMP: Should compare discret distributions.
def times_first_screen_censoring(n_samples, n_timepoints):
    """Sample times for init and final screenings."""

    print(f"Sampling {n_samples} time points.")

    x = np.arange(n_timepoints)

    p_start = st.exponnorm.pdf(x, K=8.76, loc=9.80, scale=7.07)
    t_start = np.random.choice(x, p=p_start / sum(p_start), size=n_samples)

    p_cens = st.exponweib.pdf(x, a=513.28, c=4.02, loc=-992.87, scale=707.63)
    t_cens = np.random.choice(x, p=p_cens / sum(p_cens), size=n_samples)

    # Sanity check
    assert len(t_start) == len(t_cens)

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
        t_exit = int(round(time_exit_state(age, age_max, state)))

        if t_exit >= age_max:
        	t_exit = age_max

        age = t_exit

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
    for t_a, t_b in zip(t_start, t_cens):

        d = simulate_profile(t_a, t_b, n_timepoints)

        # NOTE: Require at least 5 obs.
        if sum(d != 0) < 5:
            continue

        D.append(d)

    return np.array(D)


if __name__ == "__main__":
    # Demo run.
    np.random.seed(42)

    D = simulate_screening_histories(n_samples=1200)
    print(D)

    # Analyse histories.
    v, c = np.unique(D[D != 0], return_counts=True)
    print(v, c, c / sum(c))


    #np.save("/Users/sela/Desktop/hmm.npy", D)
    #D = np.load("/Users/sela/Desktop/hmm.npy")

    # import matplotlib.pyplot as plt 
    # plt.figure()
    # plt.imshow(D, aspect="auto")
    # plt.show()

    # A1:
    #idx = np.squeeze(np.where(np.max(D, axis=1) > 1))
    # A2:
    #t_end = np.argmax(np.cumsum(D, axis=1), axis=1)
    #idx = np.squeeze(np.where(D[range(D.shape[0]), t_end] > 2))
    
    _, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
    for i, axis in enumerate(axes.ravel()):

        x = D[idx[i], :]
        x[x == 0] = np.nan
        axis.plot(x, "o")
        #axis.set_xlim(0, 321)

        #y = np.ones(321) * np.nan
        #y[x != 0] = x[x != 0]
        #axis.plot(y, "o")

    plt.show()
 