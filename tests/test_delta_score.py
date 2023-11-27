import pytest


@pytest.mark.parametrize(
    "probabilities,correct_index,expected_delta",
    [
        [np.array([[1, 0, 0], [1, 0, 0]]), np.array([0, 0]), np.array([-1, -1])],
        [np.array([[1, 0, 0], [1, 0, 0]]), np.array([1, 1]), np.array([1, 1])],
        [np.array([[0, 1, 0], [0, 0, 1]]), np.array([1, 0]), np.array([-1, 1])],
        [
            np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]),
            np.array([0, 1]),
            np.array([0.0, 0.5]),
        ],
    ],
)
def test_delta_score(probabilities, correct_index, expected_delta):
    """Test delta score calculated as expected."""
    assert np.all(_calculate_delta(probabilities, correct_index) == expected_delta)

