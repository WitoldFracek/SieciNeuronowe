from neural_network import NeuralNetwork
from functions import relu, softmax, relu_der, sigmoid, sigmoid_der, tanh, tanh_der, relu_cutoff
import numpy as np
from binary_utils import *
from keras.datasets import mnist
import tensorflow
from tqdm import tqdm
import matplotlib.pyplot as pyl
from data_transform import make_batch, load_mnist_data, shuffle_data, show_random

np.set_printoptions(precision=3, suppress=True)

LAYER_SIZES = [(28 * 28, 50), (50, 30), (30, 20), (20, 10)]
ACT_FUNCTIONS = [tanh] * len(LAYER_SIZES)
ACT_FUNCTIONS[-1] = softmax
ACT_DERIVATIVES = [tanh_der] * len(LAYER_SIZES)
LEARNING_RATE = 0.3
ITERATIONS = 1000
BATCH_SIZE = 600
BATCH_ITERATIONS = 100


def main():
    nn = NeuralNetwork(LAYER_SIZES,
                       ACT_FUNCTIONS,
                       ACT_DERIVATIVES,
                       learning_rate=LEARNING_RATE)

    # x_train, y_train = generate_set(1000, XOR_VALUES, operator_xor)
    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle_data(x_train, y_train)

    for _ in tqdm(range(BATCH_ITERATIONS), colour='BLUE'):
        x_batch, y_batch = make_batch(BATCH_SIZE, x_train, y_train)
        for _ in range(ITERATIONS):
            nn.forward(x_batch)
            nn.backward(y_batch)
            nn.update()
    predictions = nn.forward(x_test)

    show = True
    while show:
        show_random(x_test, predictions, y_test, nbest=2)
        res = input().strip()
        if res == "exit":
            show = False
        else:
            show = True

    # print(get_random_prediction(predictions, y_test))
    # print(y_test[:, :7])
    # print()
    # print(predictions[:, :7])





    # eg = x_train.reshape((28, 28, -1)).T
    # pyl.imshow(eg[0], cmap="gray")
    # pyl.show()
    # pyl.imshow(eg[1], cmap="gray")
    # pyl.show()
    # pyl.imshow(eg[2], cmap="gray")
    # pyl.show()
    # pyl.imshow(eg[3], cmap="gray")
    # pyl.show()
    # pyl.imshow(eg[4], cmap="gray")
    # pyl.show()
    # pyl.imshow(eg[5], cmap="gray")
    # pyl.show()
    # pyl.imshow(eg[6], cmap="gray")
    # pyl.show()


if __name__ == '__main__':
    main()



