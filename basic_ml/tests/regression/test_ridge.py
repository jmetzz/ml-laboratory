import numpy as np
from numpy.testing import assert_almost_equal

from regression.ridge import feature_derivative_ridge

np.random.seed(0)


def test_feature_derivative_ridge():
    example_feature = np.array([1, 2, 3, 4, 5])
    errors = np.array([1.0, 1.5, 2.0, 1.0, 2.0])
    expected = 68.0
    actual = feature_derivative_ridge(
        errors=errors, feature=example_feature, weight=1, l2_penalty=10, feature_is_constant=False
    )
    assert_almost_equal(actual, expected)
