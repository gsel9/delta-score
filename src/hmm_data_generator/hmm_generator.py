from tqdm import tqdm

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

# TEMP: Add . before imports!
from transition import next_state, inital_state
from sojourn import sojourn_time


# TEMP: Should compare discret distributions.
def sample_init_censoring_times(n_samples):
    """Sample times for init and final screenings."""

    print(f"Sampling {n_samples} time points.")

    t_start = st.exponnorm.rvs(K=8.76, loc=9.80, scale=7.07, size=n_samples)
    t_start = t_start.astype(int)

    t_cens = st.exponweib.rvs(a=513.28, c=4.02, loc=-992.87, scale=707.63, size=n_samples)
    t_cens = t_cens.astype(int)

    # NB: Make sure t_end > t_start for all females.
    i = t_start < t_cens
    t_cens = t_cens[i]
    t_start = t_start[i]

    print(f"Kept {sum(i)} time points.")

    # Sanity checks
    assert len(t_start) == len(t_cens)

    assert np.all(t_start < t_cens)

    return t_start, t_cens


def simulate_profile(age, age_max, n_timepoints: int, missing=0) -> np.ndarray:
    """Returns a simulated screening history for one single female.
    """

    # NOTE:
    # * age, t_exit are defined on grid [t_start, t_end].
    # * period_start, period_end are defined on grid [0, n_timepoints].

    age_min = age

    x = np.ones(n_timepoints) * missing
    t_grid = np.linspace(age, age_max, n_timepoints)
    
    state = inital_state(init_age=age)

    i, period_start, period_end = 0, 0, 0
    while age < age_max:

        # Exit time from current state.
        t_exit = int(sojourn_time(age, age_max, state))

        age = age + t_exit

        # Scale to number of timepoints to cover in state vector.
        period_end = int(round(age / (age_max - age_min) * n_timepoints))

        # Update state vector.
        x[period_start:period_end] = state            

        # To avoid endless loop in case censoring.
        i += 1
        if i >= age_max:
            return x

        state = next_state(age, state)

        # Update auxillary variable.
        period_start = period_end

    return x


def simulate_screening_histories(t_start, t_cens, n_timepoints):

    D = []
    for t_a, t_b in tqdm(zip(t_start, t_cens)):

        d = simulate_profile(t_a, t_b, n_timepoints)
        
        if sum(d) == 0:
            continue

        D.append(d)

    return np.array(D)


if __name__ == "__main__":
    # Demo run.

    np.random.seed(42)

    n_samples = 120
    n_timepoints = 340

    t_start, t_cens = sample_init_censoring_times(n_samples)

    D = simulate_screening_histories(t_start, t_cens, n_timepoints)

    # Inspect
    v, c = np.unique(D[D != 0], return_counts=True)
    print(v, c, c / sum(c))

    np.save("/Users/sela/Desktop/hmm.npy", D)
    D = np.load("/Users/sela/Desktop/hmm.npy")

    import matplotlib.pyplot as plt 
    plt.figure()
    plt.imshow(D, aspect="auto")
    plt.show()

    idx = np.squeeze(np.where(np.max(D, axis=1) > 1))
    _, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 15))
    for i, axis in enumerate(axes.ravel()):

        x = D[idx[i], :]
        x[x == 0] = np.nan
        axis.plot(x, "o")
    plt.show()
