import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

from neural_networks.base.activations import sigmoid, softmax, tanh, step


def line_graph(x, y, x_title, y_title):
    """
    Draw line graph with x and y values
    :param x:
    :param y:
    :param x_title:
    :param y_title:
    :return:
    """
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


if __name__ == "__main__":
    input = np.arange(-10, 10, 0.01)
    y_value = [step(z) for z in input]
    line_graph(input, y_value, "Inputs", "Step Scores")

    y_value = sigmoid(input)
    line_graph(input, y_value, "Inputs", "Sigmoid Scores")
    pp.pprint(y_value)
    print('----')

    y_value = tanh(input)
    line_graph(input, y_value, "Inputs", "Hyperbolic Tangent Scores")
    pp.pprint(y_value)
    print('----')

    logits = np.linspace(-1, 10, num=100)
    y_value = softmax(logits)
    pp.pprint(y_value)
    line_graph(logits, y_value, "Inputs", "Softmax Scores")
