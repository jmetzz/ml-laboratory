import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from utils.data_helper import as_vector

TOY_1D = np.array(
    [
        [0.1, 1],
        [0.2, 1],
        [0.3, 1],
        [0.4, -1],
        [0.5, -1],
        [0.6, -1],
        [0.7, -1],
        [0.8, 1],
        [0.9, 1],
        [1.0, 1],
    ]
)

TOY_2D = np.array(
    [
        [2.771244718, 1.784783929, 0],
        [1.728571309, 1.169761413, 0],
        [3.678319846, 2.81281357, 0],
        [3.961043357, 2.61995032, 0],
        [2.999208922, 2.209014212, 0],
        [7.497545867, 3.162953546, 1],
        [9.00220326, 3.339047188, 1],
        [7.444542326, 0.476683375, 1],
        [10.12493903, 3.234550982, 1],
        [6.642287351, 3.319983761, 1],
    ]
)

TOY_2D_UNLABELLED = np.array(
    [
        [2.771244718, 1.784783929],
        [1.728571309, 1.169761413],
        [3.678319846, 2.81281357],
        [3.961043357, 2.61995032],
        [2.999208922, 2.209014212],
        [7.497545867, 3.162953546],
        [9.00220326, 3.339047188],
        [7.444542326, 0.476683375],
        [10.12493903, 3.234550982],
        [6.642287351, 3.319983761],
    ]
)


def play_tennis():
    features_train = np.array(
        [
            ["Sunny", "Hot", "High", "Weak"],
            ["Sunny", "Hot", "High", "Strong"],
            ["Overcast", "Hot", "High", "Weak"],
            ["Rain", "Mild", "High", "Weak"],
            ["Rain", "Cool", "Normal", "Weak"],
            ["Rain", "Cool", "Normal", "Strong"],
            ["Overcast", "Cool", "Normal", "Strong"],
            ["Sunny", "Mild", "High", "Weak"],
            ["Sunny", "Cool", "Normal", "Weak"],
        ]
    )

    labels_train = np.array(["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes"])

    features_test = np.array(
        [
            ["Rain", "Mild", "Normal", "Weak"],
            ["Sunny", "Mild", "Normal", "Strong"],
            ["Overcast", "Mild", "High", "Strong"],
            ["Overcast", "Hot", "Normal", "Weak"],
            ["Rain", "Mild", "High", "Strong"],
        ]
    )

    labels_test = np.array(["Yes", "Yes", "Yes", "Yes", "No"])
    return features_train, labels_train, features_test, labels_test


def linear_forward_test_case(num_elements: int = 3, num_features: int = 2, seed: int = 43):
    """
       X = np.array([[-1.02387576, 1.12397796],
    [-1.62328545, 0.64667545],
    [-1.74314104, -0.59664964]])
       W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
       b = np.array([[1]])
    """
    np.random.seed(seed)

    data = np.random.randn(num_elements, num_features)
    weights = np.random.randn(1, num_elements)
    bias = np.random.randn(1, 1)

    return data, weights, bias


def l_model_forward_test_case(num_elements: int = 3, num_features: int = 2, seed: int = 43):
    """
       X = np.array([[-1.02387576, 1.12397796],
    [-1.62328545, 0.64667545],
    [-1.74314104, -0.59664964]])
       parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
           [-1.07296862,  0.86540763, -2.3015387 ]]),
    'W2': np.array([[ 1.74481176, -0.7612069 ]]),
    'b1': np.array([[ 0.],
           [ 0.]]),
    'b2': np.array([[ 0.]])}
    """
    np.random.seed(seed)
    data = np.random.randn(num_elements, num_features)

    weights_1 = np.random.randn(num_elements - 1, num_elements)
    bias_1 = np.random.randn(num_elements - 1, 1)

    weights_2 = np.random.randn(1, num_elements - 1)
    bias_2 = np.random.randn(1, 1)

    parameters = {"W1": weights_1, "b1": bias_1, "W2": weights_2, "b2": bias_2}

    return data, parameters


class MNISTLoader:
    """A helper class to load the MNIST image data.

    For details of the data structures that are returned,
    see the doc strings for ``load_data`` and ``load_data_wrapper``.

    The original data set is available at: http://yann.lecun.com/exdb/mnist/
    """

    @staticmethod
    def load_data(
        path: str,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray],]:
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
        file_obj = gzip.open(path, "rb")
        train, validation, test = pickle.load(file_obj, encoding="latin1")
        file_obj.close()
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


def amazon():
    filename = "reviews_Video_Games_5.json"
    train_summary = []
    train_review_text = []
    train_labels = []

    test_summary = []
    test_review_text = []
    test_labels = []

    with Path(filename).open() as file_obj:
        for (idx, line) in enumerate(file_obj):
            data = json.loads(line)

            if data["overall"] == 3:
                continue
            if data["overall"] == 4 or data["overall"] == 5:
                label = 1
            elif data["overall"] == 1 or data["overall"] == 2:
                label = 0
            else:
                raise Exception("Unexpected value " + str(data["overall"]))

            summary = data["summary"]
            review_text = data["reviewText"]

            if idx % 10 == 0:
                test_summary.append(summary)
                test_review_text.append(review_text)
                test_labels.append(label)
            else:
                train_summary.append(summary)
                train_review_text.append(review_text)
                train_labels.append(label)

    return (train_summary, train_review_text, train_labels), (
        test_summary,
        test_review_text,
        test_labels,
    )


def load_amazon_smaller(size=20000):
    (train_summary, train_review_text, train_labels), (
        test_summary,
        test_review_text,
        test_labels,
    ) = amazon()
    return (train_summary[:size], train_review_text[:size], np.array(train_labels[:size]),), (
        test_summary[:size],
        test_review_text[:size],
        np.array(test_labels[:size]),
    )
