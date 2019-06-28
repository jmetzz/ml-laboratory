import random
from typing import List, Tuple, Sequence

import numpy as np

from neural_networks.base import losses


class GradientDescent:
    def run(epochs, loss=losses.mse, learning_rate=0.01):
        """Find the weights wk and biases bl which minimize the cost function.


        The update rule is:
            \begin{eqnarray}
              w_k & \rightarrow & w_k' = w_k-\eta \frac{\partial C}{\partial w_k} \\
              b_l & \rightarrow & b_l' = b_l-\eta \frac{\partial C}{\partial b_l}.
            \end{eqnarray}

        """

    pass


class StochasticGradientDescent:
    pass

    def evaluate(self, test_data):
        pass
