from neural_network import NeuralNetwork
from utils import relu, softmax, relu_der
import numpy as np
from binary_utils import *

np.set_printoptions(precision=3, suppress=True)

layer_sizes = [(2, 2)]  # [(2, 10), (10, 5), (5, 2)]
act_functions = [relu] * len(layer_sizes)
act_functions[-1] = softmax
act_derivatives = [relu_der] * len(layer_sizes)


def main():
    nn = NeuralNetwork(layer_sizes,
                       act_functions,
                       act_derivatives,
                       learning_rate=0.05)
    x, y = generate_set(100, OR_VALUES, operator_or)
    print(x[..., :6])
    print(y[..., 6])
    for i in range(1000):
        nn.forward(x)
        nn.backward(y)
        nn.update()

    print(nn.forward(x)[..., :6])


if __name__ == '__main__':
    main()


