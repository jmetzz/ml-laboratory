import numpy as np


def L1(yhat, y):
    """Calculates the L1 loss value between 2 vectors.

    The L1 loss function is defined as:
        \begin{align*}
            & L_1(\hat{y}, y) = \sum_{i=0}^m|y^{(i)} - \hat{y}^{(i)}|
        \end{align*}

    :param yhat: vector of size m containing the predicted labels
    :param y: vector of size m containing the true labels
    :return: the value of the L1 loss function
    """
    return np.sum(np.abs(y - yhat))


def L2(yhat, y):
    """Calculates the L1 loss value between 2 vectors.

    L2 loss is defined as
        \begin{align*}
            & L_2(\hat{y},y) = \sum_{i=0}^m(y^{(i)} - \hat{y}^{(i)})^2
        \end{align*}

    :param yhat: vector of size m containing the predicted labels
    :param y: vector of size m containing the true labels
    :return: the value of the L2 loss function
    """
    return np.sum((y - yhat) ** 2)
