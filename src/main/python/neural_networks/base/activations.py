from typing import Sequence

import numpy as np


def step(z: float) -> float:
    return 1.0 if z >= 0.0 else 0.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Calculate the sigmoid activation function

    The sigmoid function take any range real number
    and returns the output value which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.
    The output will thus be:  np.exp(x)=(e^x1, e^x2,...,e^xn)

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> sigmoid(input)

        >>> [sigmoid(x) for x in input]
    """
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z: np.ndarray) -> np.ndarray:
    """Calculate the tangent activation function

    Takes any range real number and returns the output value
    which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> tanh(input)

        >>> [tanh(x) for x in input]

    """
    # tanh(z) = 2σ(2z) − 1
    return 2 * np.array(sigmoid(z)) - 1


def hypertang(z: np.ndarray) -> np.ndarray:
    """Calculate the hyper tangent activation function

    Takes any range real number and returns the output value
    which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> hypertang(input)

        >>> [hypertang(x) for x in input]

    """
    exp = np.exp(2 * z)
    return (exp - 1) / (exp + 1)


def softmax(z: np.ndarray) -> np.ndarray:
    """"Calculate the softmax activation function

    Takes any range real number and returns
    the probabilities range with values between 0 and 1,
    and the sum of all the probabilities will be equal to one.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    :param z: a numpy matrix of shape (n, m)
    :return s: a numpy matrix equal to the softmax of x, of shape (n,m)

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> softmax(input)

        >>> [softmax(x) for x in input]
    """
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp, axis=1, keepdims=True)
    return z_exp / z_sum


if __name__ == "__main__":
    a = np.random.rand(3, 2)
    print(sigmoid(a))
