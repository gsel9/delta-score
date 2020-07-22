import numpy as np

from utils import age_group_idx, lambda_sr, p_init_state


def inital_state(init_age: int) -> int:
    """Sample a state at first screening."""
    
    return np.random.choice([1, 2, 3, 4], p=p_init_state[age_group_idx(init_age)])       


def legal_transitions(current_state: int, age_group_idx: int) -> np.ndarray:
    """Filter intensities for shifts from the current state."""

    # Transition intensities for the given age group.
    lambdas = lambda_sr[age_group_idx]
    
    # Censoring.
    if current_state == 0:
        return np.asarray([0])

    # N0 -> L1/D4.
    if current_state == 1:
        return np.array([lambdas[0], lambdas[5]])
    
    # L1 -> N0/H2/D4.
    if current_state == 2:
        return np.array([lambdas[3], lambdas[1], lambdas[6]])

    # H2 -> H2/C3/D4
    if current_state == 3:
        return np.array([lambdas[4], lambdas[2], lambdas[7]])

    # C3 -> D4
    if current_state == 4:
        return np.asarray([lambdas[8]])

    raise ValueError(f"Invalid current state: {current_state}")


def next_state(age_exit: int, current_state: int, censoring: int = 0) -> int:
    """Returns next female state."""

    lambdas = legal_transitions(current_state, age_group_idx(age_exit))

    if len(lambdas) == 1:
        return censoring

    # N0 -> L1/D4 (censoring)
    if current_state == 1:
        return np.random.choice((2, censoring), p=lambdas / sum(lambdas))

    # L1 -> N0/H2/D4
    if current_state == 2:
        return np.random.choice((1, 3, censoring), p=lambdas / sum(lambdas))
    
    # H2 -> L1/C3/D4
    if current_state == 3:
        return np.random.choice((2, 4, censoring), p=lambdas / sum(lambdas))

    if current_state == censoring:
        return censoring
    
    raise ValueError(f"Invalid current state: {current_state}")


if __name__ == '__main__':
    print(next_state(16, 2))
