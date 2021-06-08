import numpy as np


def logistic_regression(features, target, weights, iterations):
    for _ in range(iterations):
        weights -= np.dot(((1.0 / (1.0 + np.exp(-target * np.dot(features, weights))) - 1.0) * target), features)
    return weights
