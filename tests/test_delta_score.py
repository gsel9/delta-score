import numpy as np
from delta_score import compute_delta_score, compute_sample_coverage
from hypothesis import given, strategies
from hypothesis.extra.numpy import arrays
from sklearn.preprocessing import OneHotEncoder


def test_delta_score_min():
    # test delta = -1 if p_correct = 1

    y_true = np.array([[0, 1]])
    p_pred = np.array([[0.0, 1.0]])

    assert np.isclose(compute_delta_score(y_true, p_pred), -1.0)


def test_delta_score_max():
    # test max(delta) = 1 if p_correct = 0

    # ohe = OneHotEncoder()
    # y_true = ohe.fit_transform(y_true.reshape(-1, 1)).toarray()

    y_true = np.array([[0, 1]])
    p_pred = np.array([[1.0, 0]])

    assert np.isclose(compute_delta_score(y_true, p_pred), 1.0)


def test_delta_score_mean():
    # test delta = 0 if p_correct = 0.5

    y_true = np.array([[0, 1]])
    p_pred = np.array([[0.5, 0.5]])

    assert np.isclose(compute_delta_score(y_true, p_pred), 0.0)


@given(
    strategies.integers(min_value=5, max_value=30),
    strategies.integers(min_value=10, max_value=100),
    strategies.data(),
)
def test_sample_coverage(n_thresholds, n_rows, data):
    # test bound to 0-1

    y_true = data.draw(
        arrays(
            int,
            n_rows,
            elements=strategies.integers(min_value=0, max_value=1),
        )
    )

    p_pred = data.draw(
        arrays(
            float,
            n_rows,
            elements=strategies.floats(min_value=0, max_value=1, allow_nan=False),
        )
    )

    ohe = OneHotEncoder()
    y_true = ohe.fit_transform(y_true.reshape(-1, 1)).toarray()

    p_pred = np.transpose([p_pred, 1 - p_pred])

    threshold, coverage = compute_sample_coverage(y_true, p_pred, n_thresholds)

    assert np.isclose(min(threshold), -1.0)
    assert np.isclose(max(threshold), 1.0)

    assert np.isclose(min(coverage), 0.0)
    # assert np.isclose(max(coverage), 1.0)
