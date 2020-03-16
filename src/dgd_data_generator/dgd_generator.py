"""
Algorithms for simulating matrices that exhibit the same characteristics as Norwegian cervical cancer screening data.
"""

import numpy as np


def simulate_float_from_named_basis(basis_name, N=38000, T=321, K=5, domain=[1, 4], random_state=42):

    if not(random_state is None):
        np.random.seed(random_state)

    if basis_name == 'simple_peaks':
        V = np.empty(shape=(T, K))

        centers = np.linspace(70, 170, K)

        x = np.linspace(0, T, T)
        for i_k in range(K):
            V[:, i_k] = 1 + 3.0 * np.exp(-5e-4*(x - centers[i_k])**2)

        shape=1.0
        scale=1.0

    elif basis_name == 'hard_peaks':
        V = np.empty(shape=(T, K))

        centers = np.linspace(70, 170, K)

        x = np.linspace(0, T, T)
        for i_k in range(K):
            V[i_k] = 1 + 3.0 * np.exp(-5e-4*(x - centers[i_k]**2))

        shape=1.0
        scale=10.0

    U = np.random.gamma(shape, scale, size=(N, K))

    M_unscaled = U@V.T

    M = domain[0] + (M_unscaled - np.min(M_unscaled))/(np.max(M_unscaled) - np.min(M_unscaled))*(domain[1] - domain[0])

    return M


def simulate_integer_from_float(
    X_float_unscaled,
    integer_parameters,
    return_float=False,
    random_state=None
):
    """Simulation of integer data from floats.

    Parameters
    ----------
    X_float_unscaled       : Scores.
    integer_parameters   : Parameters for the simulation.
        output_domain    : Subset of the integers included in the output.
        kernel_parameter : Parameter used in the pmf.
    return_float : Return input.
    seed         : Replication of results.

    Returns
    ----------
    res : Simulated integer X_float_unscaled 
    """
    output_domain = integer_parameters['output_domain']
    kernel_parameter = integer_parameters['kernel_parameter']

    if not(random_state is None):
        np.random.seed(random_state)

    domain_max = np.max(output_domain)
    domain_min = np.min(output_domain)

    N = X_float_unscaled.shape[0]
    T = X_float_unscaled.shape[1]
    Z = output_domain.shape[0]

    X_float_scaled = domain_min + (domain_max - domain_min)*(X_float_unscaled -
                                                             np.min(X_float_unscaled))/(np.max(X_float_unscaled) - np.min(X_float_unscaled))

    def distribution(x, dom): return np.exp(-kernel_parameter*(x - dom)**2)

    domain_repeated = np.repeat(output_domain, N).reshape((N, Z), order='F')

    X_integer = np.empty_like(X_float_scaled)

    # Initialization
    column_repeated = np.repeat(
        X_float_scaled[:, 0], 4).reshape((N, 4), order='C')
    pdf = distribution(column_repeated, domain_repeated)
    cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

    u = np.random.uniform(size=(N, 1))
    indices = np.argmax(u <= cdf, axis=1)
    X_integer[:, 0] = output_domain[indices]

    # Timestepping
    for j in range(1, T):
        column_repeated = np.repeat(
            X_float_scaled[:, j], 4).reshape((N, 4), order='C')
        pdf = distribution(column_repeated, domain_repeated)
        cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

        u = np.random.uniform(size=(N, 1))
        indices = np.argmax(u <= cdf, axis=1)
        X_integer[:, j] = output_domain[indices]

    if return_float:
        return X_integer, X_float_scaled
    else:
        return X_integer