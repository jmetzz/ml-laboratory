from typing import Tuple

import numpy as np


def step(z: float) -> float:
    return 1.0 if z >= 0.0 else 0.0


def sigmoid(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the sigmoid activation function

    The sigmoid function take any range real number
    and returns the output value which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.
    The output will thus be:  np.exp(x)=(e^x1, e^x2,...,e^xn)

    Args:
        z -- Output of the linear layer, of any shape
    Returns:
        s -- a numpy matrix equal to the sigmoid of x, of shape (n,m)
        cache -- returns Z as well, useful during backpropagation

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> sigmoid(input)

        >>> [sigmoid(x) for x in input]
    """
    return 1.0 / (1.0 + np.exp(-z)), z


def tanh(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the tangent activation function

    Takes any range real number and returns the output value
    which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.
    
    Args:
        z -- Output of the linear layer, of any shape
    Returns:
        h -- a numpy matrix equal to the tanh of x, of shape (n,m)
        cache -- returns Z as well, useful during backpropagation

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> tanh(input)

        >>> [tanh(x) for x in input]

    """
    # tanh(z) = 2σ(2z) − 1
    return 2 * np.array(sigmoid(z)) - 1, z


def hypertang(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the hyper tangent activation function

    Takes any range real number and returns the output value
    which falls in the range of 0 to 1.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Args:
        z -- Output of the linear layer, of any shape
    Returns:
        h -- a numpy matrix equal to the hyperbolic tangent of x, of shape (n,m)
        cache -- returns Z as well, useful during backpropagation
    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> hypertang(input)

        >>> [hypertang(x) for x in input]

    """
    exp = np.exp(2 * z)
    return (exp - 1) / (exp + 1), z


def softmax(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """"Calculate the softmax activation function

    Takes any range real number and returns
    the probabilities range with values between 0 and 1,
    and the sum of all the probabilities will be equal to one.

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function
    element wise, that is, in vectorized form.

    Args:
        z -- Output of the linear layer, of any shape
    Returns:
        s -- a numpy matrix equal to the softmax of x, of shape (n,m)
        cache -- returns Z as well, useful during backpropagation

    Examples:
        >>> input = np.arange(-10, 10, 1)
        >>> softmax(input)

        >>> [softmax(x) for x in input]
    """
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp, axis=1, keepdims=True)
    return z_exp / z_sum, z


def relu(z: np.ndarray) -> np.ndarray:
    """Computes the Rectified Linear Unit activation function

    Arguments:
        z -- Output of the linear layer, of any shape

    Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- returns Z as well, useful during backpropagation
    """
    return np.maximum(0, z), z


def relu_backward(dA, cache):
    """Implement the backward propagation for a single RELU unit.

    Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
        dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    """Implement the backward propagation for a single SIGMOID unit.

    Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
        dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


if __name__ == "__main__":
    a = np.random.rand(3, 2)
    print(sigmoid(a))

    a = np.random.random((3, 2)) - 0.5
    print("a")
    print(a)
    print("Relu")

    print(relu(a))
