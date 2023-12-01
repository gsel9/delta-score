import numpy as np
from sklearn.metrics import auc
from sklearn.utils import check_array


def compute_delta_score(y_true, p_pred):
    """Compute the delta scores.

    Args:
            y_true (array-like (n_samples, n_columns)): Ground truth class labels.
            p_pred (array-like (n_samples, n-columns)): Class probabilities.

    Returns:
            (array-like) the delta score values.

    """

    y_true = check_array(y_true)
    p_pred = check_array(p_pred)

    p_true = np.sum(y_true * p_pred, axis=1)
    p_max_false = np.max((1 - y_true) * p_pred, axis=1)

    return p_max_false - p_true


def compute_sample_coverage(y_true, p_pred, n_thresholds: int):
    """TODO

    Args:
            delta_scores (array-like):
            n_thresholds (int):

    Returns:
            (tuple) TODO.
    """

    delta_scores = compute_delta_score(y_true, p_pred)

    threshold = np.linspace(-1, 1, n_thresholds)
    coverage = np.ones(n_thresholds) * float(np.nan)

    for i, tau in enumerate(threshold):
        coverage[i] = sum(delta_scores < tau) / delta_scores.shape[0]

    return threshold, coverage


def plot_sample_coverage(axis, y_true, p_pred, n_thresholds: int, color: str = "C0"):
    """TODO

    Args:
            y_true ():
            p_pred ():
            n_thresholds ():

    Returns:
            (axis object) TODO.
    """

    threshold, coverage = compute_sample_coverage(y_true, p_pred, n_thresholds)

    axis.plot(threshold, coverage, color=color, linewidth=2)
    axis.fill_between(threshold, coverage, color=color, alpha=0.5)

    axis.annotate(
        "AUC = " + "{:.3f}".format(auc(threshold, coverage)),
        xy=(0.5, 0.05),
        fontsize=9,
    )

    return axis
