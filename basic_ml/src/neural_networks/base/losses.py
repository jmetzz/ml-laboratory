import numpy as np


def l1_norm(estimations, true_values):
    """Calculates the L1 loss value between 2 vectors.

    The L1 loss function is defined as:
        \begin{align*}
            & L_1(\\hat{y}, y) = \\sum_{i=0}^m|y^{(i)} - \\hat{y}^{(i)}|
        \\end{align*}

    :param estimations: vector of size m containing the predicted labels
    :param true_values: vector of size m containing the true labels
    :return: the value of the L1 loss function
    """
    return np.sum(np.abs(true_values - estimations))


def l2_norm(estimations, true_values):
    """Calculates the L1 loss value between 2 vectors.

    L2 loss is defined as
        \begin{align*}
            & L_2(\\hat{y},y) = \\sum_{i=0}^m(y^{(i)} - \\hat{y}^{(i)})^2
        \\end{align*}

    :param estimations: vector of size m containing the predicted labels
    :param true_values: vector of size m containing the true labels
    :return: the value of the L2 loss function
    """
    return np.sum((true_values - estimations) ** 2)

