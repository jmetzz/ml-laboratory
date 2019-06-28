class FeedforwardNeuralNetwork:

    def __init__(self, input_size, output_size, hidden_layers):
        '''Initializes a FNN with the given input, output and
        hidden layers configuration

        Arguments:
            input_size: integer corresponding to the number of units
                on the input layer
            output_size: integer corresponding to the number of units
                on the output layer
            hidden_layers: a dictionary of the form 'layerNumber: n units'
                corresponding to the hidden layers configuration. Each unit
                is a single perceptron

        '''
        self.input_size = input_size
        self.output_size = output_size
        self.hidden = hidden_layers
