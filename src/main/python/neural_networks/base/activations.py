from typing import Sequence

import numpy as np


def step(z: float) -> float:
    return 1.0 if z >= 0.0 else 0.0


def sigmoid(z: Sequence[float]) -> Sequence[float]:
    """Calculate the sigmoid activation function

    The sigmoid function take any range real number
    and returns the output value which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> [sigmoid(input)

        >>> [sigmoid(x) for x in input]
    """
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z: Sequence[float]) -> Sequence[float]:
    """Calculate the tangent activation function

    Takes any range real number and returns the output value
    which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> [tanh(input)

        >>> [tanh(x) for x in input]

    """
    # tanh(z) = 2σ(2z) − 1
    return 2 * np.array(sigmoid(z)) - 1


def hypertang(z: Sequence[float]) -> Sequence[float]:
    """Calculate the hyper tangent activation function

    Takes any range real number and returns the output value
    which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> [hypertang(input)

        >>> [hypertang(x) for x in input]

    """
    exp = np.exp(2 * z)
    return (exp - 1) / (exp + 1)


def softmax(z: Sequence[float]) -> Sequence[float]:
    """"Calculate the softmax activation function

    Takes any range real number and returns
    the probabilities range with values between 0 and 1,
    and the sum of all the probabilities will be equal to one.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> [softmax(input)

        >>> [softmax(x) for x in input]
    """
    return np.exp(z) / np.sum(np.exp(z), axis=0)
