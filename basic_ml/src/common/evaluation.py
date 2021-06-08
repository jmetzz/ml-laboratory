import numpy as np


def accuracy(true_labels, predictions):
    return np.sum([int(y == y_hat) for y, y_hat in zip(true_labels, predictions)]) / len(true_labels)


def precision(true_labels, predictions):
    """precision = tp / tp_fp"""
    raise NotImplementedError


def recall(true_labels, predictions):
    """recall = tp / tp_fn"""
    raise NotImplementedError


def f1_score(true_labels, predictions):
    raise NotImplementedError


def confusion_matrix(true_labels, predictions):
    raise NotImplementedError


def basic_evaluation(true_labels, prediction):
    return precision(true_labels, prediction), recall(true_labels, prediction)
