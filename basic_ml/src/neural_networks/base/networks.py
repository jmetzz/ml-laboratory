import json
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from utils import data_helper

from neural_networks.base.activations import relu, sigmoid
from neural_networks.base.costs import CrossEntropyCost, QuadraticCost
from neural_networks.base.initializers import random_gaussian, sqrt_connections_ratio


class NetworkBase:
    def __init__(self, sizes: list[int]):
        """
        The biases and weights in the Network object are all
        initialized randomly, using the Numpy np.random.randn function
        to generate Gaussian distributions with mean 0 and standard deviation 1.

        Assumes that the first layer of neurons is an input layer.

        :param sizes: the number of neurons in the respective layers.

        Example:
            >>> net = SimpleNetwork([2, 3, 1])
        """
        self.num_layers = len(sizes)
        self.layer_sizes = sizes
        self.weights, self.biases = [], []

    def update_batch(self, *kwargs):
        raise NotImplementedError("This method must be implemented in a specialization class")

    def cost_name(self):
        raise NotImplementedError("This method must be implemented in a specialization class")

    def load(self, filename):
        raise NotImplementedError("This method must be implemented in a specialization class")

    def back_propagate(self, x: np.ndarray, y: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Pass x through the network and back to calculate the gradient.

        :param x: the test example to be classified
        :param y: the true label (as an index of the neuron in the output layer
        :return: the gradient for the cost function
            as a tuple (g_biases, g_weights), where the elements
            of the tuple are layer-by-layer lists of numpy arrays.
        """
        biases_by_layers = [np.zeros(b.shape) for b in self.biases]
        weights_by_layers = [np.zeros(w.shape) for w in self.weights]

        # 1- feedforward
        # the input, x, is the activation of the first layer
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        z_vectors_by_layer = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_vectors_by_layer.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # 2- backward pass
        delta = self.calculate_delta(activations, z_vectors_by_layer, y)
        biases_by_layers[-1] = delta
        weights_by_layers[-1] = np.dot(delta, activations[-2].transpose())

        # Since python allow negative index, we use it to
        # iterate backwards on the network layers.
        # Note that layer = 1 means the last layer of neurons,
        # layer = 2 is the second-last layer, and so on.
        for layer in range(2, self.num_layers):
            z = z_vectors_by_layer[-layer]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            biases_by_layers[-layer] = delta
            weights_by_layers[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return biases_by_layers, weights_by_layers

    def feed_forward(self, input: np.ndarray) -> np.ndarray:
        """Pass the input through the network and return it's output.

        It is assumed that the input a is an (n, 1) Numpy ndarray,
        not a (n,) vector
        """
        for b, w in zip(self.biases, self.weights):
            input = sigmoid(np.dot(w, input) + b)
        return input

    def calculate_delta(self, activations, z_vectors_by_layer, y):
        return QuadraticCost.derivative(activations[-1], y) * self.sigmoid_derivative(z_vectors_by_layer[-1])

    @staticmethod
    def sigmoid_derivative(z):
        """Derivative of the sigmoid function.

        This function computes the gradient (also called the slope) of
        the sigmoid function with respect to its input x.

        It is defined as:
            sigmoid\\_derivative(x) = \\sigma'(x) = \\sigma(x) (1 - \\sigma(x))\tag{2}

        :param z: a scalar or numpy array
        :return the gradient value
        """
        sigmoid_value = sigmoid(z)
        return sigmoid_value * (1 - sigmoid_value)

    def export(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {
            "sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost_name()),
        }
        with Path(filename).open(mode="w") as file_handler:
            json.dump(data, file_handler)


class SimpleNetwork(NetworkBase):
    def __init__(self, sizes: list[int]):
        super().__init__(sizes)
        self.weights, self.biases = random_gaussian(sizes)

    def cost_name(self):
        return "Quadratic cost"

    def load(self, filename):
        raise NotImplementedError(f"TODO: implement load method in {self.__name__}")

    def sdg(
        self,
        training_data: Iterator[tuple[np.ndarray, np.ndarray]],
        epochs: int,
        eta: float = 0.01,
        batch_size: int = 100,
        test_data: Iterator[tuple[np.ndarray, Any]] = None,
        debug=False,
    ) -> None:
        """Train the neural network using mini-batch stochastic gradient descent.

        :param training_data: list of tuples (X, y)
        :param epochs: number of epochs to run
        :param eta: learning rate
        :param batch_size: the size of the batch to use per iteration
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
          w_k & \rightarrow & w_k' = w_k-\frac{\\eta}{m}
          \\sum_j \frac{\\partial C_{X_j}}{\\partial w_k}\\

          b_l & \rightarrow & b_l' = b_l-\frac{\\eta}{m}
          \\sum_j \frac{\\partial C_{X_j}}{\\partial b_l},
        \\end{eqnarray}

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
            batches = [train_data[k : k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)
            if test_data:
                evaluation = self.evaluate(test_data)
                print(f"Epoch {j}: {evaluation} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_batch(self, mini_batch: list[tuple[np.ndarray, np.ndarray]], eta: float) -> None:
        """Updates the network weights and biases according to
        a single iteration of gradient descent, using just the
        training data in mini_batch and back-propagation.

        :param mini_batch: the batch of instances to process
        :param eta: the learning rate
        """
        b_hat = [np.zeros(b.shape) for b in self.biases]
        w_hat = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b_hat, delta_w_hat = self.back_propagate(x, y)
            b_hat = [nb + dnb for nb, dnb in zip(b_hat, delta_b_hat)]
            w_hat = [nw + dnw for nw, dnw in zip(w_hat, delta_w_hat)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, w_hat)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, b_hat)]

    @staticmethod
    def _print_configuration(epochs, batch_size, eta, n, n_test=None):
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"eta: {eta}")
        print(f"train set size: {n}")
        if n_test:
            print(f"test set size: {n_test}")

    def evaluate(self, data: Sequence) -> int:
        """Evaluate the network's prediction on the given test set

        :param data: the list of instances the evaluate the model
        :return: the number of test inputs correctly classified
        """
        results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


class ImprovedNetwork(NetworkBase):
    def __init__(
        self,
        layer_sizes,
        cost_function=CrossEntropyCost,
        weight_initializer=sqrt_connections_ratio,
    ):
        super().__init__(layer_sizes)
        self.num_classes = layer_sizes[-1]
        self.weights, self.biases = weight_initializer(layer_sizes)
        self.cost_function = cost_function

    def cost_name(self):
        return self.cost_function.__name__

    def load(self, filename):
        raise NotImplementedError(f"TODO: implement load method in {self.__name__}")

    def sdg(
        self,
        training_data,
        epochs,
        batch_size,
        eta,
        lmbda=0.0,
        validation_data=None,
        monitor=None,
        debug=False,
    ):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        train_data = list(training_data)
        n = len(train_data)
        validation_costs, validation_acc = [], []
        train_costs, train_acc = [], []
        if validation_data:
            validation_data = list(validation_data)

        # TODO: implement early stopping
        for epoch in range(epochs):
            if debug:
                print(f"\n****\nStarting Epoch: {epoch}")
            random.shuffle(train_data)
            batches = [train_data[k : k + batch_size] for k in range(0, n, batch_size)]
            if debug:
                batch_num = 0
                qtt_batches = len(batches)
            for batch in batches:
                if debug:
                    batch_num += 1
                    print(f"\tProcessing batch {batch_num} of {qtt_batches}")

                self.update_batch(batch, eta, lmbda, n)

            if monitor and validation_data is not None:
                # monitoring the validation accuracy more often,
                # per mini_batch maybe, or x * mini_batches
                tr_acc, tr_cost, valid_acc, valid_cost = self._monitor(monitor, lmbda, train_data, validation_data)
                train_costs.append(tr_cost)
                train_acc.append(tr_acc)
                validation_costs.append(valid_cost)
                validation_acc.append(valid_acc)

                ImprovedNetwork.print_results(epoch, valid_cost, valid_acc, tr_cost, tr_acc)

        return validation_costs, validation_acc, train_costs, train_acc

    def update_batch(
        self,
        mini_batch: list[tuple[np.ndarray, np.ndarray]],
        eta: float,
        lmbda: float,
        n: int,
    ) -> None:
        """Updates the network weights and biases according to
        a single iteration of gradient descent, using just the
        training data in mini_batch and back-propagation.

        :param mini_batch: a list of tuples, where each tuple is an instance (x, y) to process
        :param eta: the learning rate
        :param lmbda: the regularization parameter
        :param n: the total size of the training data set
        """
        b_hat = [np.zeros(b.shape) for b in self.biases]
        w_hat = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b_hat, delta_w_hat = self.back_propagate(x, y)
            b_hat = [nb + dnb for nb, dnb in zip(b_hat, delta_b_hat)]
            w_hat = [nw + dnw for nw, dnw in zip(w_hat, delta_w_hat)]
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, w_hat)
        ]

        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, b_hat)]

    def _monitor(self, monitor, lmbda, train_data, validation_data):
        train_acc = train_cost = valid_acc = valid_cost = None
        if "train_acc" in monitor or "train_cost" in monitor:
            measures = []
            if "train_acc" in monitor:
                measures.append("acc")
            if "train_cost" in monitor:
                measures.append("cost")
            transformed_data = [(x, np.argmax(y)) for x, y in train_data]
            train_acc, train_cost = self.evaluate(transformed_data, measures, lmbda)

        if "valid_acc" in monitor or "valid_cost" in monitor:
            measures = []
            if "valid_acc" in monitor:
                measures.append("acc")
            if "valid_cost" in monitor:
                measures.append("cost")
            valid_acc, valid_cost = self.evaluate(validation_data, measures, lmbda)

        return train_acc, train_cost, valid_acc, valid_cost

    def evaluate(self, test_data, measures: list = None, lmbda=5.0):
        measures = measures or ["acc"]
        data = list(test_data)
        acc = cost = None
        prediction = [self.feed_forward(x) for x, y in data]
        if "acc" in measures:
            true_labels = [y for _, y in data]
            acc = self._accuracy(zip(prediction, true_labels))
        if "cost" in measures:
            true_labels_as_vector = [data_helper.as_vector(y, self.num_classes) for _, y in data]
            cost = self._total_cost(list(zip(prediction, true_labels_as_vector)), lmbda)

        return acc, cost

    def _total_cost(self, data, lmbda):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for activations, true_label in data:
            cost += self.cost_function.evaluate(activations, true_label) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    @staticmethod
    def _accuracy(data) -> float:
        results = [(np.argmax(activations), true_label) for (activations, true_label) in data]
        return sum(int(prediction == true_label) for (prediction, true_label) in results) / len(results)

    @staticmethod
    def print_results(epoch, valid_cost, valid_acc, training_cost, training_acc):
        print(f"Epoch {epoch}:")
        if valid_acc:
            print(f"\t valid_acc: {valid_acc:.5f}")
        if valid_cost:
            print(f"\t valid_cost: {valid_cost:.5f}")
        if training_acc:
            print(f"\t training_acc: {training_acc:.5f}")
        if training_cost:
            print(f"\t training_cost: {training_cost:.5f}")
        print("-------")


class DidacticCourseraNetwork:
    def forward(self, features, parameters):
        """Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
            X -- data, numpy array of shape (input size, number of examples)
            parameters -- output of initialize_parameters_deep()

        Returns:
            AL -- last post-activation value
            caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """
        caches = []
        activations = features
        num_layers = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for layer in range(1, num_layers):
            previous_activations = activations
            activations, cache = self.linear_activation_forward(
                previous_activations, parameters[f"W{layer}"], parameters[f"b{layer}"], activation_func=relu
            )
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        linear_activation_values, cache = self.linear_activation_forward(
            activations, parameters[f"W{num_layers}"], parameters[f"b{num_layers}"], activation_func=sigmoid
        )
        caches.append(cache)
        if linear_activation_values.shape != (1, features.shape[1]):
            raise Exception("Shape mismatch when processing forward step.")
        return linear_activation_values, caches

    def linear_activation_forward(self, previous_activation_values, weights, biases, activation_func=relu):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Args:
            previous_activation_values: activations from previous layer (or input data)
                (size of previous layer, number of examples)
            weights: a numpy array of shape (size of current layer, size of previous layer)
            biases: numpy array of shape (size of the current layer, 1)
            activation_func -- the activation to be used in this layer

        Returns:
            activation_values: the output of the activation function, also called the post-activation value
            cache: a python tuple containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        pre_activations, linear_cache = self.linear_forward(previous_activation_values, weights, biases)
        activation_values, activation_cache = activation_func(pre_activations)

        if activation_values.shape != (weights.shape[0], previous_activation_values.shape[1]):
            raise Exception("Shape mismatch when processing linear activation forward function.")

        cache = (linear_cache, activation_cache)
        return activation_values, cache

    def linear_forward(self, activations, weight_matrix, bias_vector):
        """
        Implement the linear part of a layer's forward propagation.

        Args:
            activations: activations from previous layer (or input data): (size of previous layer, number of examples)
            weight_matrix: numpy array of shape (size of current layer, size of previous layer)
            bias_vector: numpy array of shape (size of the current layer, 1)

        Returns:
            tuple with the pre-activation parameter values (Z) and cache,
            a python tuple containing "activations", "weights" and "biases";
            stored for computing the backward pass efficiently
        """
        pre_activations = np.dot(weight_matrix, activations) + bias_vector

        if pre_activations.shape != (weight_matrix.shape[0], activations.shape[1]):
            raise Exception("Shape mismatch when processing linear forward function.")

        return pre_activations, (activations, weight_matrix, bias_vector)
