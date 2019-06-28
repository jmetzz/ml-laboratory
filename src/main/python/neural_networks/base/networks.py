import random
from typing import List, Tuple, Iterator, Any, Sequence

import numpy as np

from neural_networks.base.activations import sigmoid


class Network:

    def __init__(self, sizes: List[int]):
        """
        The biases and weights in the Network object are all
        initialized randomly, using the Numpy np.random.randn function
        to generate Gaussian distributions with mean 0 and standard deviation 1.

        Assumes that the first layer of neurons is an input layer.


        :param sizes: the number of neurons in the respective layers.

        Example:
            >>> net = Network([2, 3, 1])
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def sdg(self,
            training_data: Iterator[Tuple[np.ndarray, np.ndarray]],
            epochs: int,
            eta: float = 0.01,
            batch_size: int = 100,
            test_data: Iterator[Tuple[np.ndarray, Any]] = None,
            debug=False) -> None:
        """Train the neural network using mini-batch stochastic gradient descent.

        :param training_data: list of tuples (X, y)
        :param epochs: number of epochs to run
        :param eta: learning rate
        :param batch_size: the size of the batch to use per iteration
        :param loss: the cost function
        :param test_data: is provided then the network will be evaluated against
               the test data after each epoch, and partial progress printed out.
        :param debug: prints extra information
        :return:

        Estimates the gradient ∇C minimizing the cost function

        Estimates the gradient ∇C by computing ∇Cx for a small sample
        of randomly chosen training inputs. By averaging over this
        small sample it turns out that we can quickly get a
        good estimate of the true gradient ∇C.

        The update rule is:

        \begin{eqnarray}
          w_k & \rightarrow & w_k' = w_k-\frac{\eta}{m}
          \sum_j \frac{\partial C_{X_j}}{\partial w_k}\\

          b_l & \rightarrow & b_l' = b_l-\frac{\eta}{m}
          \sum_j \frac{\partial C_{X_j}}{\partial b_l},
        \end{eqnarray}

        where the sums are over all the training examples Xj in
        the current mini-batch.
        """
        train_data = list(training_data)
        n = len(train_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        if debug:
            self._print_configuration(epochs, batch_size, eta, n, n_test)

        for j in range(epochs):
            random.shuffle(train_data)

            batches = [train_data[k: k + batch_size] for k in
                       range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)
            if test_data:
                evaluation = self.evaluate(test_data)
                print(f"Epoch {j}: {evaluation} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    @staticmethod
    def _print_configuration(epochs, batch_size, eta, n, n_test=None):
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"eta: {eta}")
        print(f"train set size: {n}")
        if n_test:
            print(f"test set size: {n_test}")

    def update_batch(self, mini_batch: List[Tuple[np.ndarray, np.ndarray]],
                     eta: float) -> None:
        """Updates the network weights and biases according to
        a single iteration of gradient descent, using just the
        training data in mini_batch and back-propagation.

        :param mini_batch: the batch of instances to process
        :param eta: the learning rate
        """
        b_hat = [np.zeros(b.shape) for b in self.biases]
        w_hat = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b_hat, delta_w_hat = self.backpropagate(x, y)
            b_hat = [nb + dnb for nb, dnb in zip(b_hat, delta_b_hat)]
            w_hat = [nw + dnw for nw, dnw in zip(w_hat, delta_w_hat)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, w_hat)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, b_hat)]

    def evaluate(self, test_data: Sequence) -> int:
        """Evaluate the network's prediction on the given test set

        :param test_data:
        :return: the number of test inputs correctly classified
        """
        results = [(np.argmax(self.feed_forward(x)), y)
                   for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

    def feed_forward(self, input: np.ndarray) -> np.ndarray:
        """Pass the input through the network and return it's output.

        It is assumed that the input a is an (n, 1) Numpy ndarray,
        not a (n,) vector
        """
        for b, w in zip(self.biases, self.weights):
            input = sigmoid(np.dot(w, input) + b)
        return input

    def backpropagate(self, x: np.ndarray, y: float) \
            -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Pass x through the network and back to calculate the gradient.

        :param x: the test example to be classified
        :param y: the true label (as an index of the neuron in the output layer
        :return: the gradient for the cost function
            as a tuple (g_biases, g_weights), where the elements
            of the tuple are layer-by-layer lists of numpy arrays.
        """
        biases_by_layers = [np.zeros(b.shape) for b in self.biases]
        weights_by_layers = [np.zeros(w.shape) for w in self.weights]

        ## 1- feedforward
        # the input, x, is the activation of the first layer
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        z_vectors_by_layer = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_vectors_by_layer.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        ## 2- backward pass
        delta = self.calculate_delta(activations, z_vectors_by_layer, y)
        biases_by_layers[-1] = delta
        weights_by_layers[-1] = np.dot(delta, activations[-2].transpose())

        # Since python allow negative index, we use it to
        # iterate backwards on the network layers.
        # Note that layer = 1 means the last layer of neurons,
        # layer = 2 is the second-last layer, and so on.
        for layer in range(2, self.num_layers):
            z = z_vectors_by_layer[-layer]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            biases_by_layers[-layer] = delta
            weights_by_layers[-layer] = np.dot(delta, activations[
                -layer - 1].transpose())

        return biases_by_layers, weights_by_layers

    def calculate_delta(self, activations, z_vectors_by_layer, y):
        return self.cost_derivative(activations[-1], y) * \
               self.sigmoid_prime(z_vectors_by_layer[-1])

    @staticmethod
    def cost_derivative(output_activations, y):
        """Return the vector of partial derivatives

        \partial C_x / \partial a for the output activations."""
        return output_activations - y

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z) * (1 - sigmoid(z))
