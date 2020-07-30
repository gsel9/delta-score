"""
"""

import numpy as np 

from utils import age_partitions, age_group_idx
from transition import legal_transitions


def kappa_0(age, current_state, t) -> float:

    l = age_group_idx(age + t)

    s_l = sum(legal_transitions(current_state, l))

    return -1.0 * t * s_l


def kappa_1(age, current_state, t) -> float:

    k = age_group_idx(age)
    _, tau_kp = age_partitions[k]

    l = age_group_idx(age + t)
    tau_l, _ = age_partitions[l]

    s_k = (tau_kp - age) * sum(legal_transitions(current_state, k))    
    s_l = (age - tau_l) * sum(legal_transitions(current_state, l))
    
    return (-1.0 * s_k) - s_l


def kappa_m(age, current_state, m) -> float:
    
    k = age_group_idx(age)

    tau_km, tau_kmp = age_partitions[k + m - 1]

    s_km = (tau_kmp - tau_km) * sum(legal_transitions(current_state, k + m - 1))

    return -1.0 * s_km


def kappa(age, current_state, t, i) -> float:

    if i == 0:
        return kappa_0(age, current_state, t)
    
    if i == 1:
        return kappa_1(age, current_state, t)
    
    return kappa_m(age, current_state, i)


def eval_cdf(age, t, k, s):

    n = age_group_idx(age + t) - k + 1

    return 1.0 - np.exp(sum([kappa(age, s, t, i) for i in range(n)]))


def sojourn_time(u, k, age, age_max, s) -> np.ndarray:
    # NOTE: Time is considered relative to age. 
        
    t = 1
    t_max = age_max - age

    cdf = eval_cdf(age, t, k, s)

    while cdf <= u:

        t += 1

        # NOTE: t, t_max should be <int>.
        if t >= t_max:
            return t_max - 1

        cdf = eval_cdf(age, t, k, s)

    return t


def time_exit_state(age: int, age_max: int, s: int) -> float:
    """Returns the amount of time a female spends in the current state."""

    # Need t > 0.
    if age_max - age < 1:
        return age_max

    # Corollary 1: step 1
    u = np.random.uniform()

    # Step 2
    k = age_group_idx(age)

    # Step 3
    # i) Seek t: P(T < t) approx u where t is relative to age.
    t = sojourn_time(u, k, age, age_max, s)

    # ii) Seek l: P(T < tau_l - a) < u < P(T < tau_lp - a).
    #     If t: P(T < t) approx u => t in [tau_l - a, tau_lp - a) <=> t + a in [tau_l, tau_lp).
    l = age_group_idx(t + age)

    tau_l, tau_lp = age_partitions[l] 

    # Sanity check.
    assert tau_l - age <= t and t < tau_lp - age

    # Step 4
    sum_kappa = sum([kappa(age, s, tau_l - age, i) for i in range(1, l - k + 1)])

    return (sum_kappa - np.log(1 - u)) / sum(legal_transitions(s, l)) + age
