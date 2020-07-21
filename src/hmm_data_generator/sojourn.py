"""
"""

import numpy as np 

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


def sojourn_time_cdf(age, t_lapse, s) -> np.ndarray:
    """Compute the sojourn time CDF for a given female.
    
    Args:
        age: Start age.
        dt: ???
        s: Current state.
    
    Returns:
        Sojourn time cdf over [age, age + dt].
    """

    k = age_group_idx(age)

    dt = int(t_lapse - age)

    cdf = np.zeros(dt + 1, dtype=np.float32)
    for i, t in enumerate(range(dt)):

        t = t + age

        n = age_group_idx(age + t) - k

        cdf[i + 1] = 1.0 - np.exp(sum([kappa(age, s, t, i) for i in range(n)]))

    return cdf


# TODO: Try tau_l - age and tau_l in sum_kappa formula.
def sojourn_time(age: int, age_max: int, s: int) -> float:
    """Estimate the time that will spent in a given state.
    Args:
        age: 
        age_max:  
        s: Current age.
        
    Returns:
        The amount of time a female spends in the current state.
    """
    
    # Censor the rest of the profile.
    #if current_state == 0:
    #    return age_max

    # Corollary 1: step 1
    u = np.random.uniform(low=0, high=1.0)

    # Step 2
    k = age_group_idx(age)

    # Step 3
    cdf = sojourn_time_cdf(age, age_max, s)

    # a) First time point after intersection between CDF and u, ie min(t): P(T < t) >= u.
    t_hat = np.argmax(cdf >= u)

    # b) Map t_hat to a time partition interval l: t_hat in [tau_l - a, tau_lp - a] => 
    #    t_hat + a in [tau_l, tau_lp].
    l = age_group_idx(t_hat + age)

    tau_l, _ = age_partitions[l]

    # See proof of Corollary 1 for formula.
    sum_kappa = sum([kappa(age, s, tau_l - age, i) for i in range(1, l - k)])

    # Step 4
    return (sum_kappa - np.log(1 - u)) / sum(legal_transitions(s, l))


if __name__ == '__main__':

    # NB: Need to use `normalised` age.

    import matplotlib.pyplot as plt

    cdf = sojourn_time_cdf(16, 96, 2)
    #cdf = sojourn_time_cdf(1, 80, 2)
    print(cdf)

    plt.figure()
    plt.plot(cdf)
    plt.show()

    #for a in np.linspace(16, 96, 5, int):
    #    for b in np.linspace(a, 96, 5):
    #        print(sojourn_time(a, b, 1))

