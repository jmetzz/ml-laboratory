import csv
import gzip
import json
import pickle
import random
import os
from typing import Tuple

import numpy as np
import h5py
from keras.datasets import imdb


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


def load_h5_dataset(path: str, name: str) -> Tuple:
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


def load_imdb_pickle(vocabulary_size=1000):
    """Loads the keras imdb dataset

    Since numpy version 1.16.4 sets allow_pickle to False by default,
    we need to overwrite this parameter to be able to load the dataset into memory.

    We should, obviously, reset the default parameters later.
    """

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

    # restore np.load for future normal usage
    np.load = np_load_old
    return (X_train, y_train), (X_test, y_test)


def load_amazon():
    filename = 'reviews_Video_Games_5.json'
    train_summary = []
    train_review_text = []
    train_labels = []

    test_summary = []
    test_review_text = []
    test_labels = []

    with open(filename, 'r') as f:
        for (i, line) in enumerate(f):
            data = json.loads(line)

            if data['overall'] == 3:
                next
            elif data['overall'] == 4 or data['overall'] == 5:
                label = 1
            elif data['overall'] == 1 or data['overall'] == 2:
                label = 0
            else:
                raise Exception("Unexpected value " + str(data['overall']))

            summary = data['summary']
            review_text = data['reviewText']

            if i % 10 == 0:
                test_summary.append(summary)
                test_review_text.append(review_text)
                test_labels.append(label)
            else:
                train_summary.append(summary)
                train_review_text.append(review_text)
                train_labels.append(label)

    return (train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels)


def load_amazon_smaller(size=20000):
    (train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = load_amazon()
    return (train_summary[:size], train_review_text[:size], np.array(train_labels[:size])), (
        test_summary[:size], test_review_text[:size], np.array(test_labels[:size]))


class MNISTLoader:
    """A helper class to load the MNIST image data.

      For details of the data structures that are returned,
      see the doc strings for ``load_data`` and ``load_data_wrapper``.

      The original data set is available at: http://yann.lecun.com/exdb/mnist/
    """

    @staticmethod
    def load_data(path: str) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[
            np.ndarray, np.ndarray]]:
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
        train_results = [as_vector(y, 10) for y in tr_d[1]]
        train_data = zip(train_inputs, train_results)

        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])

        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])

        return train_data, validation_data, test_data


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
