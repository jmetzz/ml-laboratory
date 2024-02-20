import math

import numpy as np
from numpy.testing import assert_almost_equal
from regression.lasso import coordinate_descent_step


def test_lasso_coordinate_descent_step():
    expected_w = 0.425558846691
    actual_w = coordinate_descent_step(
        feature_matrix=np.array(
            [[3.0 / math.sqrt(13), 1.0 / math.sqrt(10)], [2.0 / math.sqrt(13), 3.0 / math.sqrt(10)]]
        ),
        feature_idx=1,
        weights=np.array([1.0, 4.0]),
        output=np.array([1.0, 1.0]),
        l1_penalty=0.1,
    )

    assert_almost_equal(expected_w, actual_w, decimal=12)
