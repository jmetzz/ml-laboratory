import numpy as np
import pytest
from numpy.testing import assert_array_equal
from utils.data_helper import as_vector, normalize


def test_as_vector():
    expected = np.array([0, 0, 1, 0, 0]).reshape((5, 1))
    actual = as_vector(2, 5)
    assert_array_equal(actual, expected)


def test_as_vector_wrong_arguments():
    with pytest.raises(ValueError) as err:
        _ = as_vector(5, 2)
    assert "Out of bounds. idx must be included in [0, num_of_label) interval" in str(err.value)


def test_normalize():
    features = np.array([[3.0, 6.0, 9.0], [4.0, 8.0, 12.0]])
    expected_values = np.array([[0.6, 0.6, 0.6], [0.8, 0.8, 0.8]])
    expected_norms = np.array([5.0, 10.0, 15.0]).reshape((1, 3))

    actual_normalized, actual_norms = normalize(features, axis=0)

    assert_array_equal(actual_normalized, expected_values)
    assert_array_equal(actual_norms, expected_norms)
