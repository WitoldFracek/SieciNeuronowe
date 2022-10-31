from neural_network import NeuralNetwork
from utils import relu, softmax, relu_der, sigmoid, sigmoid_der
import numpy as np
from binary_utils import *

np.set_printoptions(precision=3, suppress=True)

layer_sizes = [(2, 5), (5, 2)]
act_functions = [relu] * len(layer_sizes)
act_functions[-1] = softmax
act_derivatives = [relu_der] * len(layer_sizes)


def main():
    nn = NeuralNetwork(layer_sizes,
                       act_functions,
                       act_derivatives,
                       learning_rate=0.1)
    x, y = generate_set(1000, XOR_VALUES, operator_xor)
    # x, y = generate_set_flat(1000, AND_VALUES)

    for i in range(1000):
        # print(f"Iter {i + 1}")
        a = nn.forward(x)
        nn.backward(y)
        nn.update()
    print(x[..., :6])
    print(a[..., :6])


if __name__ == '__main__':
    main()


