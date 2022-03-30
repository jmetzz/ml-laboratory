import math

import numpy as np


def simple_euclidean(instance1, instance2, dimension):
    distance = 0
    for dim in range(dimension):
        distance += pow((instance1[dim] - instance2[dim]), 2)
    return math.sqrt(distance)


def euclidean(instance1, instance2):
    """Vectorized implementation of Euclidean distance"""
    delta = instance1 - instance2
    return np.sqrt(np.sum(delta * delta))


def weighted_euclidean(instance1, instance2, weights):
    delta = instance1 - instance2
    return np.sqrt((np.sum(weights * (delta * delta))))
