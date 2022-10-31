from neural_network import NeuralNetwork
from utils import relu, softmax, relu_der, sigmoid, sigmoid_der, tanh, tanh_der, load_mnist_data
import numpy as np
from binary_utils import *
from keras.datasets import mnist
import tensorflow
from tqdm import tqdm
import matplotlib.pyplot as pyl

np.set_printoptions(precision=3, suppress=True)
np.seterr(divide='ignore', invalid='ignore')

LAYER_SIZES = [(28 * 28, 50), (50, 30), (30, 20), (20, 10)]
ACT_FUNCTIONS = [tanh] * len(LAYER_SIZES)
ACT_FUNCTIONS[-1] = softmax
ACT_DERIVATIVES = [tanh_der] * len(LAYER_SIZES)
LEARNING_RATE = 0.3
ITERATIONS = 1000


def main():
    nn = NeuralNetwork(LAYER_SIZES,
                       ACT_FUNCTIONS,
                       ACT_DERIVATIVES,
                       learning_rate=LEARNING_RATE)

    # x, y = generate_set(1000, XOR_VALUES, operator_xor)
    x_train, y_train, x_test, y_test = load_mnist_data()
    for _ in tqdm(range(ITERATIONS), colour='GREEN'):
        a = nn.forward(x_train)
        nn.backward(y_train)
        nn.update()
    print(y_train[:, :7])
    print(a[:, :7])
    eg = x_train.reshape(28, 28, -1).T
    pyl.imshow(eg[0], cmap="Greys")
    pyl.show()
    pyl.imshow(eg[1], cmap="Greys")
    pyl.show()
    pyl.imshow(eg[2], cmap="Greys")
    pyl.show()
    pyl.imshow(eg[3], cmap="Greys")
    pyl.show()
    pyl.imshow(eg[4], cmap="Greys")
    pyl.show()
    pyl.imshow(eg[5], cmap="Greys")
    pyl.show()
    pyl.imshow(eg[6], cmap="Greys")
    pyl.show()


if __name__ == '__main__':
    main()


