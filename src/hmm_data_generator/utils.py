"""
Key:
    N0 = 1
    L1 = 2
    H2 = 3
    C3 = 4
    D5 = 5
"""

import numpy as np
import pandas as pd


NUM_TIMEPOINTS = 321

# NOTE: Should be (a, n] = tuple(a, b - 1)???
age_groups = np.array([
    (16, 19),  
    (20, 24),
    (25, 29),
    (30, 34), 
    (35, 39), 
    (40, 49), 
    (50, 59), 
    (60, 120)
])

#age_partitions = ((age_groups - np.min(age_groups)) / np.max(age_groups)) * NUM_TIMEPOINTS 
age_partitions = age_groups

# Transition intensities (age group x state transition).
# TODO: Add confidence bounds and do a random sampling from this range.
lambda_sr = np.array(
    [ 
        # N0->L1   L1->H2   H2->C3  L1->N0  H2->N1  N0->D4   L1->D4   H2->D4   C3->D4
        [0.01991, 0.01665, 0.00251, 0.1771, 0.2262, 0.00002, 0.00016, 0.00241, 0.01817], 
        [0.01202, 0.02526, 0.00025, 0.1550, 0.1079, 0.00006, 0.00017, 0.00083, 0.03025],
        [0.00731, 0.04176, 0.00014, 0.1448, 0.0811, 0.00012, 0.00020, 0.00168, 0.03286],
        [0.00573, 0.04201, 0.00017, 0.1520, 0.0739, 0.00012, 0.00017, 0.00278, 0.03676],
        [0.00537, 0.03467, 0.00048, 0.1553, 0.0670, 0.00010, 0.00016, 0.00419, 0.03961],
        [0.00537, 0.02938, 0.00096, 0.1664, 0.0830, 0.00010, 0.00015, 0.00537, 0.02655],
        [0.00429, 0.02495, 0.00128, 0.1933, 0.0959, 0.00020, 0.00030, 0.00621, 0.02490],
        [0.00395, 0.03383, 0.01325, 0.2348, 0.0582, 0.00104, 0.00121, 0.01558, 0.03107] 
    ]
)


# Initial state probabilities (age group x probability initial state).
p_init_state = np.array(
    [   
        [0.93020, 0.06693, 0.00263, 0.00024],
        [0.92937, 0.06228, 0.00821, 0.00014],
        [0.93384, 0.04945, 0.01654, 0.00017],
        [0.94875, 0.03574, 0.01528, 0.00023],
        [0.95348, 0.03226, 0.01400, 0.00026],
        [0.95543, 0.03309, 0.01132, 0.00016],
        [0.96316, 0.02806, 0.00847, 0.00031],
        [0.96032, 0.02793 ,0.01134, 0.00041]
    ]
)


# TODO: np.logical_and + precomputed left and right borders.
def age_group_idx(age: int) -> int:
    """Returns index i: tau_i <= age < tau_i+1."""

    for num, (tau_p, tau_pp) in enumerate(age_partitions):

        if age <= tau_p and age < tau_pp:
            return num
    
    # NOTE: Assing last patition index to age exceeding the time interval.
    return num


def profiles_to_csv(X, path_to_file):

    df = pd.DataFrame(X)
    df.to_csv(path_to_file, index=False, header=False)
