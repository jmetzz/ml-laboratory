import csv
import gzip
import pickle
import random
from typing import Tuple

import numpy as np


def load_train_test_datasets(filename, split=0.5):
    """Retrieve the training and test randomly split of the dataset
    based on the given split value

    :param filename: full path to the file containing the data
    :param split: the value between 0 and 1 (inclusive) used to split the data
    :return: a tuple with train and test sets of instances in the dataset
    """
    training_set = []
    test_set = []
    dataset = load_dataset(filename)
    for line in range(len(dataset)):
        if random.random() < split:
            training_set.append(dataset[line])
        else:
            test_set.append(dataset[line])
    return training_set, test_set


def load_dataset(filename):
    """Loads the dataset from a file

    :param filename: full path to the file containing the data
    :return: a list of instances in the dataset
    """
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(filter(None, lines))
        ncol = len(dataset[0])
        nlin = len(dataset)
        for line in range(nlin):
            for col in range(ncol - 1):
                dataset[line][col] = float(dataset[line][col])
    return dataset


toy_labeled_dataset = [[2.771244718, 1.784783929, 0],
                       [1.728571309, 1.169761413, 0],
                       [3.678319846, 2.81281357, 0],
                       [3.961043357, 2.61995032, 0],
                       [2.999208922, 2.209014212, 0],
                       [7.497545867, 3.162953546, 1],
                       [9.00220326, 3.339047188, 1],
                       [7.444542326, 0.476683375, 1],
                       [10.12493903, 3.234550982, 1],
                       [6.642287351, 3.319983761, 1]]

toy_unlabeled_dataset = [[2.771244718, 1.784783929],
                         [1.728571309, 1.169761413],
                         [3.678319846, 2.81281357],
                         [3.961043357, 2.61995032],
                         [2.999208922, 2.209014212],
                         [7.497545867, 3.162953546],
                         [9.00220326, 3.339047188],
                         [7.444542326, 0.476683375],
                         [10.12493903, 3.234550982],
                         [6.642287351, 3.319983761]]


class MNISTLoader:
    """A helper class to load the MNIST image data.

      For details of the data structures that are returned,
      see the doc strings for ``load_data`` and ``load_data_wrapper``.
    """

    @staticmethod
    def load_data(path: str) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[
            np.ndarray, np.ndarray]]:
        """Loads the MNIST data as a tuple of training, validation, and test data.

        :return: traning_data, validation_data, test_data

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
        f = gzip.open(path, 'rb')
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
        train_results = [cls.vectorized_result(y) for y in tr_d[1]]
        train_data = zip(train_inputs, train_results)

        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])

        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])

        return train_data, validation_data, test_data

    @staticmethod
    def vectorized_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
