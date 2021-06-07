import numpy as np
from numpy import log, nan_to_num


class CrossEntropyCost:
    @staticmethod
    def evaluate(activation, true_label):
        """Return the cost associated with an output activation and desired output
        ``y``.

        The cross-entropy is positive number.
        Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(nan_to_num(-true_label * log(activation) - (1 - true_label) * log(1 - activation)))

    @staticmethod
    def delta(z, activation, true_label):
        """Computes the error delta from the output layer.

        Note that the  parameter ``z`` is not used by the method.
        It is included in the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return activation - true_label


class QuadraticCost:
    @staticmethod
    def evaluate(activation, true_label):
        """Return the cost associated with an output activation and desired true label (``y``)."""
        return 0.5 * np.linalg.norm(activation - true_label) ** 2

    @staticmethod
    def delta(z, activation, true_label):
        """Computes the error delta from the output layer."""
        return (activation - true_label) * QuadraticCost.sigmoid_prime(z)

    @staticmethod
    def derivative(output_activations, y):
        """Return the vector of partial derivatives

        \partial C_x / \partial a for the output activations."""
        return output_activations - y
