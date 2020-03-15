import numpy as np
import pandas as pd


# Resolution: 3 mo /pts
# Age range: 16 - 60 (53 y, Jerome)
TIME_RANGE = [
    range(0, 17),
    range(17, 37),
    range(37, 57),
    range(57, 77),
    range(77, 97),
    range(97, 137),
    range(137, 177),
    range(177, 321)
]


def transition_probas():
    """State transition probabilities."""
    # Age range x transition state
    P_trans = np.array(
        [
            [0.19910, 0.01665, 0.00251, 0.1771, 0.2262],
            [0.01202, 0.02526, 0.00025, 0.1550, 0.1079],
            [0.00731, 0.04176, 0.00014, 0.1448, 0.0811],
            [0.00573, 0.04201, 0.00017, 0.1520, 0.0739],
            [0.00537, 0.03467, 0.00048, 0.1553, 0.0670],
            [0.00537, 0.02938, 0.00096, 0.1664, 0.0830],
            [0.00429, 0.02495, 0.00128, 0.1933, 0.0959],
            [0.00395, 0.03383, 0.01325, 0.2348, 0.0582]
        ]
    )
    return P_trans


def init_screening_probas():
    """State probabilities at initial screening."""
    
    # Age range x risk state. Row stochastic.
    P_init = np.array(
        [
            [0.93020, 0.06693, 0.00263, 0.00024],
            [0.92937, 0.06228, 0.00821, 0.00014],
            [0.93384, 0.04945, 0.01654, 0.00017],
            [0.94875, 0.03574, 0.01528, 0.00023],
            [0.95348, 0.03226, 0.01400, 0.00026],
            [0.95543, 0.03309, 0.01132, 0.00016],
            [0.96316, 0.02806, 0.00847, 0.00031],
            [0.96032, 0.02793, 0.01134, 0.00041]
        ]
    )
    return P_init


def get_transition_probas(path_to_file=None):
    
    s0_to_s1 = []
    s1_to_s2 = []
    s2_to_s3 = []
    s1_to_s0 = []
    s2_to_s1 = []
    
    P_trans = transition_probas()

    for num, idx in enumerate(TIME_RANGE):
        
        s0_to_s1.append(np.repeat(P_trans[num, 0], len(idx)))
        s1_to_s2.append(np.repeat(P_trans[num, 1], len(idx)))
        s2_to_s3.append(np.repeat(P_trans[num, 2], len(idx)))
        s1_to_s0.append(np.repeat(P_trans[num, 3], len(idx)))
        s2_to_s1.append(np.repeat(P_trans[num, 4], len(idx)))

    s0_to_s1 = np.concatenate(s0_to_s1)
    s1_to_s2 = np.concatenate(s1_to_s2)
    s2_to_s3 = np.concatenate(s2_to_s3)
    s1_to_s0 = np.concatenate(s1_to_s0)
    s2_to_s1 = np.concatenate(s2_to_s1)

    df_Pt = pd.DataFrame(
        np.vstack((s0_to_s1, s1_to_s2, s2_to_s3, s1_to_s0, s2_to_s1)),
        index=['s0=>s1', 's1=>s2', 's2=>s3', 's1=>s0', 's2=>s1']
    )

    if path_to_file is not None:
        df_Pt.to_csv(path_to_file)

    return df_Pt


def get_init_screening_probas(path_to_file=None):

    N1 = []
    R2 = []
    R3 = []
    C4 = []
    
    P_init = init_screening_probas()

    for num, idx in enumerate(TIME_RANGE):
        
        N1.append(np.repeat(P_init[num, 0], len(idx)))
        R2.append(np.repeat(P_init[num, 1], len(idx)))
        R3.append(np.repeat(P_init[num, 2], len(idx)))
        C4.append(np.repeat(P_init[num, 3], len(idx)))

    N1 = np.concatenate(N1)
    R2 = np.concatenate(R2)
    R3 = np.concatenate(R3)
    C4 = np.concatenate(C4)

    df_init = pd.DataFrame(np.vstack((N1, R2, R3, C4)), index=['N1', 'R2', 'R3', 'C4'])

    if path_to_file is not None:
        df_init.to_csv(path_to_file)

    return df_init


def shift_from_s1(time_index, df_Pt, states=[1, 2]):
    """Two events: P not state shift = 1 - P state shift."""

    np.random.seed(0)
    
    p2 = float(df_Pt.iloc[0, time_index])
    p1 = 1.0 - p2
    
    return int(np.random.choice(states, p=(p1, p2)))


def shift_from_s2(time_index, df_Pt, states=[1, 2, 3]):
    """Three events where P sums to one."""

    np.random.seed(0)
    
    p3 = float(df_Pt.iloc[1, time_index])
    p1 = float(df_Pt.iloc[3, time_index])
    p2 = 1 - p1 - p3
    
    return int(np.random.choice(states, p=(p1, p2, p3)))


def shift_from_s3(time_index, df_Pt, states=[2, 3, 4]):
    """Three events where P sums to one."""

    np.random.seed(0)
    
    p4 = float(df_Pt.iloc[2, time_index])
    p2 = float(df_Pt.iloc[4, time_index])
    p3 = 1 - p4 - p2
    
    return int(np.random.choice(states, p=(p2, p3, p4)))


def set_init_state(x, time_index, df_init):
    """Assings intial state according to empirical probabilities of being in a
    particular state at the time of the first screening."""

    np.random.seed(0)
    
    new_state = np.random.choice([1, 2, 3, 4], p=df_init.iloc[:, time_index])
    
    x[time_index] = int(new_state)


def _impute_trajectory(x, otrain, Ptransition, force_init=False, break_at_dropout=True):
    """
    Args:
        x: A row in X.
        otrain: A row in Otrain.
    """

    # Impute x[0] with a randomly selected value.
    if force_init:
        set_init_state(x, time_index=0)

    if break_at_dropout:
        nz = x.nonzero()[0]
    else:
        nz = np.append(x.nonzero()[0], len(x))

    for num, nz_idx in enumerate(nz[:-1]):
        
        next_nz_idx = int(nz[num + 1])
        
        # No need to impute.
        if next_nz_idx - nz_idx < 2:
            continue

        prev_state = int(x[nz_idx])
        for time_index in range(nz_idx + 1, next_nz_idx):
            
            if prev_state == 1:
                state = shift_from_s1(time_index, Ptransition)
            
            elif prev_state == 2:
                state = shift_from_s2(time_index, Ptransition)
            
            elif prev_state == 3:
                state = shift_from_s3(time_index, Ptransition)

            else:
                continue

            x[time_index] = state
            otrain[time_index] = 1

            prev_state = state


def sanity_check_impute(X, Ximp, Otrain, Otest):

    # Check test scores preserved.
    assert np.array_equal(X * Otest, Ximp * Otest)

    # Check training scores preserved.
    assert np.array_equal(X * Otrain, Ximp * Otrain)

    # Check each row contains at least one known score.
    assert sum(np.sum(X * Otrain, axis=1) == 0) == 0


def impute(X, O_train, O_test):

    p_transit = get_transition_probas()

    X_imp = X.copy()
    O_imp = O_train.copy()

    start_idx = np.argmax(O_train, axis=1)
    end_idx = np.argmax(np.cumsum(O_train, axis=1), axis=1)

    for num, (x, o) in enumerate(zip(X_imp, O_imp)):
        impute_trajectory(x, o, start_idx[num], end_idx[num], p_transit)

    X_imp[O_train.nonzero()] = X[O_train.nonzero()]
    X_imp[O_test.nonzero()] = X[O_test.nonzero()]

    return X_imp, O_imp


def impute_trajectory(x, o, start_idx, end_idx, p_transit, missing=0):

    idx = start_idx
    while idx < end_idx:

        # Propagate zeros after cancer cases.
        current_score = x[idx]
        if current_score == 0:
            idx = idx + 1

            continue

        next_score = x[idx + 1]
        if next_score == missing:
            x[idx + 1] = sample_next_score(current_score, idx + 1, p_transit=p_transit)

        idx = idx + 1

    o[start_idx:end_idx] = 1


def sample_next_score(current_score, next_idx, p_transit, num_iter=10):

    np.random.seed(0)

    scores = []
    for _ in range(num_iter):

        if current_score == 1:
            next_state = shift_from_s1(next_idx, p_transit)

        elif current_score == 2:
            next_state = shift_from_s2(next_idx, p_transit)

        elif current_score == 3:
            next_state = shift_from_s3(next_idx, p_transit)

        else:
            # Propagate zeros after cancer cases.
            next_state = 0

        scores.append(next_state)

    return np.random.choice(scores)


if __name__ == '__main__':

    from ioutils.readers import npz_to_ndarray
    
    X = npz_to_ndarray('/Users/sela/phd/data/real/4K/3p/X_train.npz')
    O_train = npz_to_ndarray('/Users/sela/phd/data/real/4K/3p/O_train_3p.npz')
    O_test = npz_to_ndarray('/Users/sela/phd/data/real/4K/3p/O_test_val_3y.npz')

    Ximputed, Otrain = impute(X, O_train, O_test)

    sanity_check_impute(X, Ximputed, O_train, O_test)
    
    X = np.save('/Users/sela/phd/data/real/4K/3p/X_train_imputed.npy', Ximputed)
    X = np.save('/Users/sela/phd/data/real/4K/3p/O_train_imputed.npy', Otrain)
