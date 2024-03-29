from collections.abc import Sequence

import numpy as np


def random_gaussian(layer_sizes: Sequence) -> tuple[list, list]:
    """Initializes the weights and biases using Gaussian random variables

    Assumes that the first layer of neurons is an input layer,
    and uses mean 0 and standard deviation 1.

    :returns a tuple with the weights list and biases list
    """
    biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
    # [2, 5, 4] -> [(2, 5), (5, 4)]
    layer_size_pairs = zip(layer_sizes[:-1], layer_sizes[1:])
    weights = [np.random.randn(y, x) for x, y in layer_size_pairs]
    return weights, biases


def sqrt_connections_ratio(layer_sizes: Sequence) -> tuple[list, list]:
    """Weights are initialized using Gaussian divided by the square root of the number of connections

    Assumes that the first layer of neurons is an input layer.

    Initializes the weights input to a neuron are initialized as
    Gaussian random variables with mean 0 and standard deviation 1
    divided by the square root of the number of connections input
    to the neuron. Also in this method we'll initialize the biases,
    using Gaussian random variables with mean 0 and standard deviation 1

    :returns a tuple with the weights list and biases list
    """
    biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
    # [2, 5, 4] -> [(2, 5), (5, 4)]
    layer_size_pairs = zip(layer_sizes[:-1], layer_sizes[1:])
    weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in layer_size_pairs]
    return weights, biases


def dnn_parameters_initializer(layer_dims: list[int]) -> dict:
    """Initialized weights using Gaussian divided by the square root of the number of connections

    Assumes that the first layer of neurons is an input layer.

    Initializes the weights input to a neuron are initialized as
    Gaussian random variables with mean 0 and standard deviation 1
    divided by the square root of the number of connections input
    to the neuron. Also in this method we'll initialize the biases,
    using Gaussian random variables with mean 0 and standard deviation 1

    Args:
        layer_dims: python array (list) containing the
                    dimensions of each layer in our network

    Returns:
        parameters, a python dictionary containing your parameters
        "W1", "b1", ..., "WL", "bL":
            Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
            bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    num_layers = len(layer_dims)  # number of layers in the network

    for layer in range(1, num_layers):
        parameters["W" + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters["b" + str(layer)] = np.zeros((layer_dims[layer], 1))
    return parameters
