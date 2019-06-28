from typing import Tuple, Any

from neural_networks.base.neurons import PerceptronNeuron


class BitOperation:
    def apply(self, inputs: Tuple[int, int]) -> Tuple[Any, Any]:
        raise NotImplementedError


class BitAdder(BitOperation):
    # The first weight is always the bias
    _weights = [3, -2, -2]

    def __init__(self):
        self.hidden_l1 = PerceptronNeuron()
        self.hidden_l2 = [PerceptronNeuron(), PerceptronNeuron(),
                          PerceptronNeuron()]
        self.out_layer = PerceptronNeuron()

    def apply(self, inputs: Tuple[int, int]) -> Tuple[Any, Any]:
        b1, b2 = inputs
        p1 = self.hidden_l1.predict([b1, b2], self._weights)
        p21 = self.hidden_l2[0].predict([b1, p1], self._weights)
        p22 = self.hidden_l2[1].predict([b2, p1], self._weights)
        carry = self.hidden_l2[2].predict([p1, p1], self._weights)
        out = self.out_layer.predict([p21, p22], self._weights)
        return out, carry


if __name__ == '__main__':
    adder = BitAdder()
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    print(f"Testing ADD operation on inputs: {inputs}")
    print("Input  -> (sum, carry)")
    for i in inputs:
        print(f"{i} -> {adder.apply(i)}")
