import random
from typing import List, Tuple, Iterator, Any, Sequence

import numpy as np

from neural_networks.base.activations import sigmoid
from neural_networks.base.costs import CrossEntropyCost
from neural_networks.base.initializers import random_gaussian, sqrt_connections_ration


class NetworkBase:

    def update_batch(self, *kwargs):
        raise NotImplementedError("This method must be implemented in a specialization class")

    def back_propagate(self, x: np.ndarray, y: float) \
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


class SimpleNetwork(NetworkBase):

    def __init__(self, sizes: List[int]):
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
        self.sizes = sizes
        self.weights, self.biases = random_gaussian(sizes)

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
            batches = [train_data[k: k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)
            if test_data:
                evaluation = self.evaluate(test_data)
                print(f"Epoch {j}: {evaluation} / {n_test}")
            else:
                print(f"Epoch {j} complete")

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
            delta_b_hat, delta_w_hat = self.back_propagate(x, y)
            b_hat = [nb + dnb for nb, dnb in zip(b_hat, delta_b_hat)]
            w_hat = [nw + dnw for nw, dnw in zip(w_hat, delta_w_hat)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, w_hat)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, b_hat)]

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
        results = [(np.argmax(self.feed_forward(x)), y)
                   for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


class ImprovedNetwork(NetworkBase):

    def __init__(self, layer_sizes, cost=CrossEntropyCost, weight_initializer=sqrt_connections_ration):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights, self.biases = weight_initializer(layer_sizes)
        self.cost = cost

    def sdg(self,
            training_data,
            epochs,
            batch_size,
            eta,
            lmbda=0.0,
            validation_data=None,
            measures=None):
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
        eval_loss, eval_acc = [], []
        train_loss, train_acc = [], []
        if validation_data:
            validation_data = list(validation_data)

        for epoch in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k: k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, eta, lmbda, n)

            training_acc, training_loss, valid_acc, valid_loss = self.evaluate(lmbda, measures, train_data,
                                                                               validation_data)
            train_loss.append(training_loss)
            train_acc.append(training_acc)
            eval_loss.append(valid_loss)
            eval_acc.append(valid_acc)

            ImprovedNetwork.print_results(epoch, valid_loss, valid_acc, training_loss, training_acc,
                                          len(validation_data), n)

        return eval_loss, eval_acc, train_loss, train_acc

    def update_batch(self,
                     mini_batch: List[Tuple[np.ndarray, np.ndarray]],
                     eta: float,
                     lmbda: float,
                     n: int) -> None:
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
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, w_hat)]

        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, b_hat)]

    def evaluate(self, lmbda, measures, train_data, validation_data):
        training_acc = training_loss = valid_acc = valid_loss = None
        if 'train_acc' in measures:
            training_acc = self.evaluate_acc(train_data, convert=True)
        if 'train_loss' in measures:
            training_loss = self.evaluate_loss(train_data, lmbda, convert=False)
        if 'valid_acc' in measures:
            valid_acc = self.evaluate_acc(validation_data, convert=False)
        if 'valid_loss' in measures:
            valid_loss = self.evaluate_loss(validation_data, lmbda, convert=True)
        return training_acc, training_loss, valid_acc, valid_loss

    def evaluate_acc(self, data: Sequence, convert: False) -> float:
        """Evaluate the network's prediction on the given test set

        :param data:
        :return: the number of test inputs correctly classified
        """
        if convert:
            prediction = [(self.feed_forward(x), self.as_vector(y)) for x, y in data]
        else:
            prediction = [(self.feed_forward(x), y) for x, y in data]
        return self.accuracy(prediction)

    def evaluate_loss(self, data: Sequence, lmbda: float, convert: False) -> float:
        """Evaluate the network's prediction on the given test set

        :param data:
        :param lmbda:
        :return: the number of test inputs correctly classified
        """
        if convert:
            prediction = [(self.feed_forward(x), self.as_vector(y)) for x, y in data]
        else:
            prediction = [(self.feed_forward(x), y) for x, y in data]
        return self.total_cost(prediction, lmbda)

    def total_cost(self, data, lmbda):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for activations, true_label in data:
            cost += self.cost.evaluate(activations, true_label) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    @staticmethod
    def accuracy(data):
        results = [(np.argmax(activations), true_label) for (activations, true_label) in data]
        return sum(int(prediction == true_label) for (prediction, true_label) in results)

    @staticmethod
    def print_results(epoch, valid_loss, valid_acc, training_loss, training_acc, valid_size, training_size):
        print(f"Epoch {epoch}:")
        if valid_acc: print(f"\t valid_acc: {valid_acc} / {valid_size}")
        if valid_loss: print(f"\t valid_loss: {valid_loss:.5f}")
        if training_acc: print(f"\t training_acc: {training_acc} / {training_size}")
        if training_loss: print(f"\t training_loss: {training_loss:.5f}")
        print("-------")

    @staticmethod
    def as_vector(index):
        """Return a 10-dimensional unit vector with a 1.0 in the j'th position
        and zeroes elsewhere.  This is used to convert a digit (0...9)
        into a corresponding desired output from the neural network.

        """
        e = np.zeros((10, 1))
        e[index] = 1.0
        return e
