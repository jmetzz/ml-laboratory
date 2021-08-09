import numpy as np


def feature_derivative_ridge(errors: np.ndarray,
                             feature: np.ndarray,
                             weight: np.ndarray,
                             l2_penalty: float,
                             feature_is_constant: bool = False) -> np.ndarray:
    """
    Compute the derivative of the regression cost function w.r.t L2 norm.

    If feature_is_constant is True,
    derivative is twice the dot product of errors and feature
    Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight

    """
    l2_term_derivative = 0.0 if feature_is_constant else 2 * l2_penalty * weight
    derivative = 2 * np.dot(errors, feature) + l2_term_derivative
    return derivative


def predict_output(feature_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Create a regression prediction vector by using dot product

    Args:
        feature_matrix: a 2D numpy array with features as columns
        weights: a 1D numpy array of the corresponding feature weights

    Returns:
        a 1D numpy array with the respective model prediction
    """
    predictions = np.dot(feature_matrix, weights)
    return predictions


def ridge_regression_gradient_descent(feature_matrix,
                                      output,
                                      initial_weights,
                                      step_size: float = 1e-2,
                                      l2_penalty: float = 1e-2,
                                      max_iterations=1e3,
                                      tolerance: float = 1e-3):
    weights = np.array(initial_weights)  # make sure it's a numpy array
    iteration = 0
    converged = False
    while not converged and iteration < max_iterations:
        iteration += 1
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output

        # fixme: vectorize this loop
        for feature_idx in range(len(weights)):
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # (Remember: when i=0, you are computing the derivative of the constant!)
            derivative = feature_derivative_ridge(errors=errors,
                                                  feature=feature_matrix[:, feature_idx],
                                                  weight=weights[feature_idx],
                                                  l2_penalty=l2_penalty,
                                                  feature_is_constant=(feature_idx == 0))
            weights[feature_idx] -= step_size * derivative
        gradient_magnitude = np.sqrt(np.power(-2 * errors, 2).sum())
        converged = gradient_magnitude < tolerance
    return weights
