import random

import numpy as np
from utils.data_loader import load_from_csv


def as_vector(idx, num_of_labels):
    """Return a num_of_labels-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.

    This is used to convert a digit (0...9) into a corresponding desired output
    from the neural network.

    """
    if idx >= num_of_labels:
        raise ValueError("Out of bounds. idx must be included in [0, num_of_label) interval")
    vector = np.zeros((num_of_labels, 1))
    vector[idx] = 1.0
    return vector


def normalize(data: np.ndarray, axis=0, order=2) -> tuple[np.ndarray, np.ndarray]:
    """Normalize all elements in a vector or matrix

    This operation is done by dividing each row vector of x by its norm, i.e.,
        x / || x ||

    After the operation each row of x should be a vector of unit length.

    Args:
        data: a numpy array to normalize
        order: Order of the norm (see numpy.linalg.norm documentation)
        axis: specifies the axis of the input array along which
            to compute the vector norms. 0 for columns, and 1 for lines.

    Return:
        - a normalized version of the input array
        - the norm array
    """
    norms = np.linalg.norm(data, ord=order, axis=axis, keepdims=True)
    return (
        data / norms,
        norms,
    )


def train_test_split(filename: str, split=0.5) -> tuple:
    """Retrieve the training and test randomly split of the dataset
    based on the given split value

    :param filename: full path to the file containing the data
    :param split: the value between 0 and 1 (inclusive) used to split the data
    :return: a tuple with train and test sets of instances in the dataset
    """
    training_set = []
    test_set = []
    content = load_from_csv(filename)
    for _, value in enumerate(content):
        if random.random() < split:
            training_set.append(value)
        else:
            test_set.append(value)
    return training_set, test_set
