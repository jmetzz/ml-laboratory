import csv
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np


def as_vector(idx, num_of_labels):
    """Return a num_of_labels-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.

    This is used to convert a digit (0...9) into a corresponding desired output
    from the neural network.

    """
    vector = np.zeros((num_of_labels, 1))
    vector[idx] = 1.0
    return vector


def normalize(data: np.ndarray, axis=1, order=2) -> np.ndarray:
    """Normalize all elements in a vector or matrix

    This operation is done by dividing each row vector of x by its norm, i.e.,
        x / || x ||

    After the operation each row of x should be a vector of unit length.

    :param order: Order of the norm (see numpy.linalg.norm documentation)
    :param axis: specifies the axis of the input array along which
        to compute the vector norms. 0 for columns, and 1 for lines.
    :param data: a numpy array to normalize
    :return: a normalized version of the input array
    """
    return data / np.linalg.norm(data, ord=order, axis=axis, keepdims=True)


def split_set(content, ratio: float, labels=None) -> tuple:
    temp_data = dict(content)
    part_1 = {}
    part_2 = {}
    for label, data in temp_data.items():
        if labels and label not in labels:
            continue

        random.shuffle(data)
        split_point = int(len(data) * ratio)
        part_1[label] = data[:split_point]
        part_2[label] = data[split_point:]

    return part_1, part_2


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


def load_json_file(filename, labels=None, split_ratio=0.1):
    with Path(filename).open() as json_file:
        content = json.load(json_file)
    return split_set(content, split_ratio, labels)


def load_from_csv(filename: str) -> list:
    """Loads the dataset from a file

    :param filename: full path to the file containing the data
    :return: a list of instances in the dataset
    """
    with Path(filename).open() as csv_file:
        lines = csv.reader(csv_file)

    content = list(filter(None, lines))
    ncol = len(content[0])
    nlin = len(content)
    for line in range(nlin):
        for col in range(ncol - 1):
            content[line][col] = float(content[line][col])
    return content


def load_from_h5(path: str, name: str) -> tuple:
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
