import pprint as pp

import matplotlib.pyplot as plt
import numpy as np

from neural_networks.base.activations import sigmoid, softmax, step, tanh


def line_graph(x_values, y_values, x_title, y_title):
    """
    Draw line graph with x and y values
    :param x_values:
    :param y_values:
    :param x_title:
    :param y_title:
    :return:
    """
    plt.axhline(0, color="gray")
    plt.axvline(0, color="gray")
    plt.plot(x_values, y_values)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


if __name__ == "__main__":
    x_data = np.arange(-10, 10, 0.01)
    y_data = [step(z) for z in x_data]
    line_graph(x_data, y_data, "Inputs", "Step Scores")

    y_data = sigmoid(x_data)
    line_graph(x_data, y_data, "Inputs", "Sigmoid Scores")
    pp.pprint(y_data)
    print("----")

    y_data = tanh(x_data)
    line_graph(x_data, y_data, "Inputs", "Hyperbolic Tangent Scores")
    pp.pprint(y_data)
    print("----")

    logits = np.linspace(-1, 10, num=100)
    y_data = softmax(logits)
    pp.pprint(y_data)
    line_graph(logits, y_data, "Inputs", "Softmax Scores")
