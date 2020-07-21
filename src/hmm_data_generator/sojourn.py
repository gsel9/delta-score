"""
"""

import numpy as np 

# TEMP: Add . before imports!
# TODO: Add config module with constants.
from utils import lambda_sr, age_partitions, age_group_idx
from transition import legal_transitions


def kappa_0(age, current_state, t) -> float:
    # NOTE: `age` is running female age and not age at first screening.

    return -1.0 * t * sum(legal_transitions(current_state, age_group_idx(age + t)))


def kappa_1(age, current_state, t) -> float:
    # NOTE: `tau` is normalised time grid.

    k = age_group_idx(age)
    l = age_group_idx(age + t)

    tau_l, _ = age_partitions[l]
    _, tau_kp = age_partitions[k]

    s_k = sum(legal_transitions(current_state, k))    
    s_l = sum(legal_transitions(current_state, l))

    return -1.0 * (tau_kp - age) * s_k - (age - tau_l) * s_l
    

def kappa_m(age, current_state, m) -> float:
    
    k = age_group_idx(age)

    tau_k, tau_kp = age_partitions[k + m]

    return -1.0 * (tau_kp - tau_k) * sum(legal_transitions(current_state, k + m - 1))


def kappa(age, current_state, t, m) -> float:
    
    if m == 0:
        return kappa_0(age, current_state, t)
    
    if m == 1:
        return kappa_1(age, current_state, t)
    
    return kappa_m(age, current_state, m)


def sojourn_time_cdf(t_start, t_end, current_state) -> np.ndarray:
    """Compute the sojourn time CDF for a given female."""

    k = age_group_idx(t_start)

    cdf = np.zeros(t_end - t_start + 1, dtype=np.float32)
    for i, dt in enumerate(range(t_start, t_end)):

        n = age_group_idx(t_start + dt) - k

        cdf[i + 1] = 1.0 - np.exp(sum([kappa(t_start, current_state, dt, i) for i in range(n)]))

    return cdf


# TODO:
def sojourn_time(age: int, age_max: int, current_state: int, tol=1e-16) -> float:
    """Estimate the time that will spent in a given state.
    Args:
        age: 
        age_max:  
        current_state: 
        
    Returns:
        The amount of time a female spends in the current state.
    """
    
    # Censor the rest of the profile.
    if current_state == 0:
        return age_max

    sojourn_cdf = sojourn_time_cdf(age, age_max, current_state)

    # Corollary 1: step 1
    u = np.random.uniform(low=tol, high=1.0 - tol)

    # Step 2
    k = age_group_idx(age)

    # Step 3


    """
    t_lower = np.squeeze(np.where(u > sojourn_cdf))
    if np.ndim(t_lower) > 0:
        t_lower = t_lower[-1]

    # Shift time point by age to satisfy l: P(T < tau_l - a) < u.
    l = age_group_idx(age + t_lower)

    # NB: Adjust to Python count logic and start sum from 1.
    sum_k = sum([kappa(age, current_state, age + t_lower, i) for i in range(1, l - k)])
    sum_p = sum(legal_transitions(current_state, lambda_sr[l, :]))
    """

    # Step 4.
    return (sum_k - np.log(1 - u)) / sum_p


if __name__ == '__main__':

    # NB: Need to use `normalised` age.

    import matplotlib.pyplot as plt

    cdf = sojourn_time_cdf(16, 96, 3)
    print(cdf)

    plt.figure()
    plt.plot(cdf)
    plt.show()
    
    #print(sojourn_time(16, 36, 1))
    #for a in np.linspace(16, 96, 5, int):
    #    for b in np.linspace(a, 96, 5, int):
    #        print(sojourn_time(a, b, 1))
            #print(a, b, sojourn_time(a, b, 1))
    
