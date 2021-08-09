from math import sqrt

import numpy as np
from numpy.linalg import inv
from numpy import dot, power
from numpy.random import default_rng


def multiple_linear_regression_closed_form(observations, y_values):
    """
    This operation is very expensive. Calculating the inverse is N^3.
    There are better solutions for this problem.

    Notes:
    - Num of observations must be greater than the number of features
    - linearly independent
    - w and y are vectors

    RSS(w) = -2H^T (y - Hw)

    Solve for w:
        RSS(w_hat) = 0
        -2H^T (y - H * w_hat) = 0
        -2H^T * y + 2H^T * H * w_hat = 0
        H^T * H * w_hat = H^T * y
        inv(H^T * H) * H^T * H * w_hat = H^T * y * inv(H^T * H)

        # inv(H^T * H) * H^T * H --> I
        # thus:

        I * w_hat = H^T * y * inv(H^T * H)
        w_hat = inv(H^T * H) * H^T * y
    """
    return dot(
        inv(observations.T @ observations),
        observations.T.dot(y_values)
    )


def random_init(size, seed=12345678903141592653589793):
    rng = default_rng(seed)
    return rng.standard_normal(size)


def gradient_descent_loop_form(feature_matrix,
                               y_values,
                               step_size: float = 1e-2,
                               tolerance: float = 1e-3,
                               max_iter: int = 1e2,
                               initializer=random_init):
    weights = initializer(feature_matrix.shape[1])
    converged = False
    iteration = 0
    while not converged and iteration < max_iter:
        predictions = predict_output(feature_matrix, weights)
        residual_errors = predictions - y_values
        gradient_sum_squares = 0

        for i in range(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(residual_errors, feature_matrix[:, i])
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += derivative ** 2
            weights[i] -= step_size * derivative
        # compute the square-root of the gradient sum of squares to get the gradient magnitude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        converged = gradient_magnitude < tolerance
        iteration += 1
    return weights


def predict_output(feature_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Predict the regression value for N input elements

    Args:
        feature_matrix: a numpy matrix containing the features as columns
        weights: the model weights as a numpy array

    Returns:
        the predictions vector as a np.array
    """
    return np.dot(feature_matrix, weights.T)


def feature_derivative(feature, errors):
    return 2 * np.dot(errors, feature)


def gradient_descent_vectorized_form(feature_matrix,
                                     y_values,
                                     step_size: float = 1e-2,
                                     tolerance: float = 1e-3,
                                     max_iter: int = 1e2,
                                     initializer=random_init):
    weights = initializer(feature_matrix.shape[1])
    converged = False
    iteration = 0

    while not converged and iteration < max_iter:
        residual_errors = y_values - feature_matrix.dot(weights.T)
        partials = -2 * feature_matrix.T.dot(residual_errors)
        weights += step_size * partials
        gradient_magnitude = np.sqrt(power(partials, 2).sum())
        converged = gradient_magnitude < tolerance
        iteration += 1
    return weights


if __name__ == "__main__":
    # first feature is the constant 1 (intercept)
    feature_matrix = np.array([[1, 10, 3],
                               [1, 2.5, 7],
                               [1, 15, 8],
                               [1, 6, 2],
                               [1, 67, 37],
                               ])

    y_values = np.array([-1, 0.2, 0.9, 3, 0.6])
    step_size = 7e-12
    tolerance = 2.5e7

    closed_w = multiple_linear_regression_closed_form(feature_matrix, y_values)
    loop_w = gradient_descent_loop_form(feature_matrix, y_values, step_size, tolerance)
    vectorized_w = gradient_descent_vectorized_form(feature_matrix, y_values, step_size, tolerance)

    print(closed_w)
    print(loop_w)
    print(vectorized_w)
