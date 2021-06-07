from neural_networks.base.activations import step
from neural_networks.base.neurons import Neuron


class SingleLayer:
    def __init__(self, bias, dimension, epochs=50, learning_rate=0.001, activation_function=step):
        self.neuron = Neuron(activation_function)
        self.bias = bias
        self.dimension = dimension
        self.eta = learning_rate
        self.epochs = epochs

    def train(self, data):
        self.weights = [0.0] * (self.dimension + 1)
        for e in range(self.epochs):
            sum_error = 0.0
            for x in data:
                # print(weights) # uncomment the check the change in weights
                predicted = self.neuron.predict(x[:-1], self.weights)
                error = x[-1] - predicted
                sum_error += error ** 2
                for i in range(self.dimension):
                    self.weights[i + 1] += self.eta * error * x[i]
            print(">epoch=%d, learning rate=%.3f, error=%.3f" % (e, self.eta, sum_error))
        return self.weights

    def predict(self, x):
        # TO DO
        pass
