import numpy as np


def as_vector(idx, num_of_labels):
    """Return a num_of_labels-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.

    This is used to convert a digit (0...9) into a corresponding desired output
    from the neural network.

    """
    e = np.zeros((num_of_labels, 1))
    e[idx] = 1.0
    return e


def normalize(data: np.ndarray, axis=1, ord=2) -> np.ndarray:
    """Normalize all elements in a vector or matrix

    This operation is done by dividing each row vector of x by its norm, i.e.,
        x / || x ||

    After the operation each row of x should be a vector of unit length.

    :param ord: Order of the norm (see numpy.linalg.norm documentation)
    :param axis: specifies the axis of the input array along which
        to compute the vector norms. 0 for columns, and 1 for lines.
    :param data: a numpy array to normalize
    :return: a normalized version of the input array
    """
    return data / np.linalg.norm(data, ord=ord, axis=axis, keepdims=True)
