import csv
import gzip
import os
import pickle
import random
from typing import Tuple

import h5py

import numpy as np
from utils.data_helper import as_vector


def train_test_split(filename, split=0.5):
    """Retrieve the training and test randomly split of the dataset
    based on the given split value

    :param filename: full path to the file containing the data
    :param split: the value between 0 and 1 (inclusive) used to split the data
    :return: a tuple with train and test sets of instances in the dataset
    """
    training_set = []
    test_set = []
    content = load_from_csv(filename)
    for line in range(len(content)):
        if random.random() < split:
            training_set.append(content[line])
        else:
            test_set.append(content[line])
    return training_set, test_set


def load_from_csv(filename):
    """Loads the dataset from a file

    :param filename: full path to the file containing the data
    :return: a list of instances in the dataset
    """
    with open(filename, "rt") as csvfile:
        lines = csv.reader(csvfile)
        content = list(filter(None, lines))
        ncol = len(content[0])
        nlin = len(content)
        for line in range(nlin):
            for col in range(ncol - 1):
                content[line][col] = float(content[line][col])
    return content


def load_from_h5(path: str, name: str) -> Tuple:
    train = h5py.File(os.path.join(path, f"train_{name}.h5"), "r")
    x_orig = np.array(train["train_set_x"][:])  # your train set features
    y_orig = np.array(train["train_set_y"][:])  # your train set labels
    y_orig = y_orig.reshape((1, y_orig.shape[0]))

    test = h5py.File(os.path.join(path, f"test_{name}.h5"), "r")
    test_x_orig = np.array(test["test_set_x"][:])  # your test set features
    test_y_orig = np.array(test["test_set_y"][:])  # your test set labels
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))

    classes = np.array(test["list_classes"][:])  # the list of classes

    return x_orig, y_orig, test_x_orig, test_y_orig, classes


class MNISTLoader:
    """A helper class to load the MNIST image data.

    For details of the data structures that are returned,
    see the doc strings for ``load_data`` and ``load_data_wrapper``.

    The original data set is available at: http://yann.lecun.com/exdb/mnist/
    """

    @staticmethod
    def load_data(
        path: str,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Loads the MNIST data as a tuple of training, validation, and test data.

        :return: training_data, validation_data, test_data

            The ``training_data`` is returned as a tuple with two entries.
            The first entry contains the actual training images.  This is a
            numpy ndarray with 50,000 entries.  Each entry is, in turn, a
            numpy ndarray with 784 values, representing the 28 * 28 = 784
            pixels in a single MNIST image.

            The second entry in the ``training_data`` tuple is a numpy ndarray
            containing 50,000 entries.  Those entries are just the digit
            values for the corresponding images contained in the first
            entry of the tuple.

            The ``validation_data`` and ``test_data`` are similar, except
            each contains only 10,000 images.

            This is a nice data format, but for use in neural networks it's
            helpful to modify the format of the ``training_data`` a little.
            That's done in the wrapper function ``load_data_wrapper()``, see
            below.
        """
        f = gzip.open(path, "rb")
        train, validation, test = pickle.load(f, encoding="latin1")
        f.close()
        return train, validation, test

    @classmethod
    def load_data_wrapper(cls, path: str):
        """Loads the wrapped MNIST data (tuple of training, validation, and test data)

        The wrapped format is more convenient for use in the implementation
        of neural networks.

        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.
            ``x`` is a 784-dimensional numpy.ndarray containing the input image.
            ``y`` is a 10-dimensional numpy.ndarray representing the unit vector
                  corresponding to the correct digit for ``x``.

        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case,
            ``x`` is a 784-dimensional numpy.ndarray containing the input image.
            ``y`` is the corresponding classification, i.e., the digit values (integers)
                  corresponding to ``x``.

        This means slightly different formats for
        the training data and the validation/test data. These formats
        turn out to be the most convenient for use in our neural network
        code."""
        tr_d, va_d, te_d = cls.load_data(path)

        train_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        train_results = [as_vector(y, 10) for y in tr_d[1]]
        train_data = zip(train_inputs, train_results)

        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])

        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])

        return train_data, validation_data, test_data
