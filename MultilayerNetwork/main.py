from neural_network import NeuralNetwork
from utils import relu, softmax, relu_der, sigmoid, sigmoid_der, tanh, tanh_der
import numpy as np
from binary_utils import *
from keras.datasets import mnist
import tensorflow
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

LAYER_SIZES = [(2, 2), (2, 2)]
ACT_FUNCTIONS = [relu] * len(LAYER_SIZES)
ACT_FUNCTIONS[-1] = softmax
ACT_DERIVATIVES = [relu_der] * len(LAYER_SIZES)
LEARNING_RATE = 0.5
ITERATIONS = 10_000


def main():
    nn = NeuralNetwork(LAYER_SIZES,
                       ACT_FUNCTIONS,
                       ACT_DERIVATIVES,
                       learning_rate=LEARNING_RATE,
                       init_weights_range=1)

    x, y = generate_set(1000, XOR_VALUES, operator_xor)
    for _ in tqdm(range(ITERATIONS), colour='GREEN'):
        a = nn.forward(x)
        nn.backward(y)
        nn.update()
    print(x[..., :7])
    print(a[..., :7])


if __name__ == '__main__':
    main()


