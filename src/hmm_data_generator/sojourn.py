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


def sojourn_time_cdf(age, age_max, s) -> np.ndarray:
    """Compute the sojourn time CDF for a given female.
    
    Args:
        age: Start age.
        age_max: End of time scale.
        s: Current state.
    
    Returns:
        Sojourn time CDF over [age, age_max].
    """

    time_lapse = int(age_max - age) + 1

    k = age_group_idx(age)

    cdf = []
    for t in range(1, time_lapse):

        # NOTE: Adjust to Python count logic (range() terminates at n - 1).
        n = age_group_idx(age + t) - k + 1

        cdf.append(1.0 - np.exp(sum([kappa(age, s, t, i) for i in range(n)])))
    
    return np.array(cdf)


def sojourn_time(age: int, age_max: int, s: int) -> float:
    """Estimate the time that will spent in a given state.
    Args:
        age: 
        age_max:  
        s: Current age.
        
    Returns:
        The amount of time a female spends in the current state.
    """
    
    # Female is censored.
    if s == 0:
        return age_max

    # Too close to state vector completion.
    if age_max - age < 2:
        return age_max

    # Corollary 1: step 1
    u = np.random.uniform()

    # Step 2
    k = age_group_idx(age)

    # Step 3: 
    cdf = sojourn_time_cdf(age, age_max, s)

    # Solve t: P(T < t) approx u where t is relative to age.
    t = np.argmax(u < cdf)

    # Exceeds CDF domain.
    if t == 0:
        t = age_max

    # Seek l: P(T < tau_l - a) < u < P(T < tau_lp - a).
    # If t: P(T < t) approx u => t in [tau_l - a, tau_lp - a) <=> t + a in [tau_l, tau_lp).
    l = age_group_idx(t + age)
    tau_l, _ = age_partitions[l]

    # Sanity check.
    #assert tau_l - age < t and age < tau_lp - age

    # Step 4
    n = age_group_idx(tau_l - age) - k + 1

    sum_kappa = sum([kappa(age, s, tau_l - age, i) for i in range(1, n)])

    return (sum_kappa - np.log(1 - u)) / sum(legal_transitions(s, l))


if __name__ == '__main__':
    """
    import matplotlib.pyplot as plt

    cdf1 = sojourn_time_cdf(16, 96, 2)
    cdf2 = sojourn_time_cdf(50, 96, 2)

    print(cdf1)
    print(cdf2)

    _, axes = plt.subplots(ncols=2, figsize=(15, 5))
    axes[0].plot(cdf1)
    axes[1].plot(cdf2)
    plt.show()
    """

    # Should be shorter for higher current states.
    print(sojourn_time(20, 70, 1))
    print(sojourn_time(20, 70, 2))
    print(sojourn_time(20, 70, 3))
    print(sojourn_time(20, 70, 4))
