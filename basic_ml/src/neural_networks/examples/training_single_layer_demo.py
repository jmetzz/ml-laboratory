from neural_networks.architectures.SingleLayer import SingleLayer
from neural_networks.base.activations import sigmoid
from neural_networks.base.neurons import PerceptronNeuron

dataset = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1],
]


def main():
    neuron = PerceptronNeuron()
    # The values of bias and weights are typically set randomly and then updated using gradient descent
    w = [-0.1, 0.20653640140000007, -0.23418117710000003]
    for row in dataset:
        prediction = neuron.predict(row, w)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    print("-----------------------------")
    rna = SingleLayer(-0.1, 2, 50, 0.01, sigmoid)

    nweights = rna.train(dataset)
    print("Network weights: ")
    print(nweights)


if __name__ == "__main__":
    main()
