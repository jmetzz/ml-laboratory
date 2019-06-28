"""Demo of how to train a neural net to recognize digits from 0 to 9

The training data for the network consists of many 28 by 28 pixel images
of scanned handwritten digits, and so the input layer contains
784=28×28 neurons (784-dimensional vector).

Each entry in the vector represents the grey value for a single pixel in the image.
Thus, the input pixels are greyscale, with a value of 0.0 representing white,
a value of 1.0 representing black, and in between values representing
gradually darkening shades of grey.

The output layer of the network contains 10 neurons, which are numbered
from 0 through 9. The neuron which neuron has the highest activation value
represents the network's guess (classification).
For example, if a particular training image, x, depicts a 6,
then y(x)=[0,0,0,0,0,0,1,0,0,0]^T is the desired output from the network.


The loss function is the 'mean squared error' (or just MSE):
\begin{eqnarray}  C(w,b) \equiv
  \frac{1}{2n} \sum_x \| y(x) - a\|^2.
\tag{6}\end{eqnarray}


w  denotes the collection of all weights in the network, b all the biases,
n is the total number of training inputs, a is the vector of outputs from
the network when x is input, and the sum is over all training inputs, x.
Of course, the output a depends on x, w and b.

So the aim of the training algorithm will be to minimize the cost C(w,b)
as a function of the weights and biases. In other words, we want to find
a set of weights and biases which make the cost as small as possible.
For that the training phase uses the algorithm known as gradient descent.

"""

from neural_networks.base import networks
from utils.data_helper import MNISTLoader

if __name__ == '__main__':
    training_set, validation_set, test_set = MNISTLoader.load_data_wrapper(
        '../data/processed/MNIST/mnist.pkl.gz')
    net = networks.Network([784, 20, 10])
    net.sdg(training_set, epochs=20, eta=2.5, batch_size=20, test_data=test_set)
