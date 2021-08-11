import csv
import os
from pathlib import Path

import h5py
import numpy as np


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
