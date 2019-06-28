from typing import List

from neural_networks.base.activations import step, sigmoid


class Neuron:
    """Base class for neurons

    Implementation classes should override the activate method.
    """

    def __init__(self, activation):
        self._activation = activation

    def predict(self, x: List, w: List) -> float:
        return self._activation(self.sum(x, w))

    @staticmethod
    def sum(x: List[float], w: List[float]) -> float:
        # The first weight is always the bias as it is standalone
        # and not responsible for a specific input value.
        output = w[0]
        for i in range(len(w) - 1):
            output += w[i + 1] * x[i]
        return output


class PerceptronNeuron(Neuron):
    """A Neuron implementation that outputs 0 or 1

    The perceptron neuron uses the step function as activation.
    """

    def __init__(self):
        super(PerceptronNeuron, self).__init__(step)


class SigmoidNeuron(Neuron):
    """A Neuron implementation that outputs σ(w⋅x+b)

     where σ is called the sigmoid function, sometimes also called
     the logistic function, and outputs values between 0 and 1, both inclusive.
    """

    def __init__(self):
        super(SigmoidNeuron, self).__init__(sigmoid)
