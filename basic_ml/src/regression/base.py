import numpy as np
from numpy.random import default_rng


def residual_sum_of_squares(predictions, y_values):
    """
    Calculate the Residual Sum of Squares (RSS)

    The sum of the squares of the residuals and the residuals is
    the difference between the predicted output and the true output.

    Returns:
        the RSS value
    """
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = predictions - y_values
    # square the residuals and add them up
    return np.sum(residuals * residuals)


def predict_output(feature_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Predict the regression value for N input elements

    Args:
        feature_matrix: a 2D numpy matrix containing the features as columns
        weights: a 1D numpy array of the corresponding feature weights

    Returns:
        a 1D numpy array with the respective model prediction
    """
    return np.dot(feature_matrix, weights.T)


# def predict_output(feature_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
#     """
#     Create a regression prediction vector by using dot product
#
#     Args:
#         feature_matrix: a 2D numpy array with features as columns
#         weights: a 1D numpy array of the corresponding feature weights
#
#     Returns:
#         a 1D numpy array with the respective model prediction
#     """
#     predictions = np.dot(feature_matrix, weights)
#     return predictions


def random_init(size, seed=12345678903141592653589793):
    rng = default_rng(seed)
    return rng.standard_normal(size)
