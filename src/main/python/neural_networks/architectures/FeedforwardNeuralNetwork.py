class FeedforwardNeuralNetwork:

    def __init__(self, inputSize, outputSize, hiddenLayers):
        '''Initializes a FNN with the given input, output and
        hidden layers configuration

        Arguments:
            inputSize: integer corresponding to the number of units
                on the input layer
            outputSize: integer corresponding to the number of units
                on the output layer
            hiddenLayers: a dictionary of the form 'layerNumber: n units'
                corresponding to the hidden layers configuration. Each unit
                is a single perceptron

        '''
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hidden = hiddenLayers
