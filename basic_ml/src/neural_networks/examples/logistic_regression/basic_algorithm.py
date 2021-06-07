import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class IterativeBinaryLogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def run(self, train_data, eta):
        train_data = list(train_data)
        n = len(train_data)
        b_slope = 0
        weight_slopes = np.zeros((self.num_features, 1))
        cost = 0

        # TODO: implement 'm' iterations

        # Calculate the Logistic Regression Derivatives
        # 1-step of Gradient Descent

        for x, y in train_data:
            # TODO: implement the dot operation with for loops
            z = np.dot(self.weights, x) + self.bias
            a = sigmoid(z)
            cost += -(y * np.log(a) + (1 - y) * np.log(1 - a))
            z_slope = a - y
            b_slope += z_slope
            for i in range(self.num_features):
                weight_slopes[i] += x[i] * z_slope
        cost /= n
        b_slope /= n
        weight_slopes = [w / n for w in weight_slopes]

        # Update the weights and bias
        for i in range(self.num_features):
            self.weights[i] -= eta * weight_slopes[i]
        self.bias -= eta * b_slope


class VectorizedBinaryLogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def run(self, train_data, eta):
        train_data = list(train_data)
        n = len(train_data)
        b_slope = 0
        weight_slopes = np.zeros((self.num_features, 1))
        cost = 0

        # TODO: implement 'm' iterations

        # Calculate the Logistic Regression Derivatives
        # 1-step of Gradient Descent
        # TODO: vectorize the loop
        for x, y in train_data:
            z = np.dot(self.weights, x) + self.bias
            a = sigmoid(z)
            cost += -(y * np.log(a) + (1 - y) * np.log(1 - a))
            z_slope = a - y
            b_slope += z_slope
            weight_slopes += x * z_slope
        cost /= n
        b_slope /= n
        weight_slopes /= n

        # Update the weights and bias
        # TODO: vectorize
        for i in range(self.num_features):
            self.weights[i] -= eta * weight_slopes[i]
        self.bias -= eta * b_slope

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
