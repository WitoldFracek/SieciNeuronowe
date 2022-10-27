from neural_network import NeuralNetwork
from utils import relu, softmax, relu_der, sigmoid, sigmoid_der
import numpy as np
from binary_utils import *

np.set_printoptions(precision=3, suppress=True)

layer_sizes = [(2, 2), (2, 3), (3, 2)]
act_functions = [relu] * len(layer_sizes)
act_functions[-1] = softmax
act_derivatives = [relu_der] * len(layer_sizes)


def main():
    nn = NeuralNetwork(layer_sizes,
                       act_functions,
                       act_derivatives,
                       learning_rate=0.01)
    x, y = generate_set(10, AND_VALUES, operator_and)
    # x, y = generate_set_flat(100, AND_VALUES)
    # print(x[..., :6])
    # print(y[..., :6])
    for i in range(10):
        print(f"Iter {i + 1}")
        nn.forward(x)
        nn.backward(y)
        nn.update()
        print()

    print(nn.forward(x)[..., :6])


if __name__ == '__main__':
    main()


