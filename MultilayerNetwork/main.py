from neural_network import NeuralNetwork
from functions import *
import numpy as np
from binary_utils import *
from keras.datasets import mnist
import tensorflow
from tqdm import tqdm
import matplotlib.pyplot as pyl
from data_transform import make_batch, load_mnist_data, shuffle_data, show_random

np.set_printoptions(precision=3, suppress=True)

# LAYER_SIZES = [(28 * 28, 70), (70, 50), (50, 10)]  # accuracy 91%
LAYER_SIZES = [(28 * 28, 50), (50, 40), (40, 30), (30, 10)]
ACT_FUNCTIONS = [tanh] * len(LAYER_SIZES)
ACT_FUNCTIONS[-1] = softmax
ACT_DERIVATIVES = [tanh_der] * len(LAYER_SIZES)
LEARNING_RATE = 0.3
ITERATIONS = 400
BATCH_SIZE = 100
BATCH_NUMBER = 100
STANDARD_DEV = 0.01
MEAN = 0


def main():
    nn = NeuralNetwork(LAYER_SIZES,
                       ACT_FUNCTIONS,
                       ACT_DERIVATIVES,
                       learning_rate=LEARNING_RATE,
                       standard_dev=STANDARD_DEV,
                       mean=MEAN)

    # x_train, y_train = generate_set(1000, XOR_VALUES, operator_xor)
    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle_data(x_train, y_train)

    training_sets = [make_batch(BATCH_SIZE, x_train, y_train) for _ in range(BATCH_NUMBER)]

    for _ in range(ITERATIONS):
        for x_batch, y_batch in training_sets:
            nn.forward(x_batch)
            nn.backward(y_batch)
            nn.update()
        # x_exmpl, y_exmpl = training_sets[0]
        # err = nn.get_error(mean_squared_error, x_exmpl, y_exmpl, output_transform=lambda pred: np.argmax(pred, axis=0))
        # print(err)
    # for _ in tqdm(range(BATCH_ITERATIONS), colour='BLUE'):
    #     x_batch, y_batch = make_batch(BATCH_SIZE, x_train, y_train)
    #     for _ in range(ITERATIONS):
    #         nn.forward(x_batch)
    #         nn.backward(y_batch)
    #         nn.update()

    predictions = nn.forward(x_test)

    values = np.argmax(predictions, axis=0)
    print(np.mean(values == y_test))

    show = True
    while show:
        show_random(x_test, predictions, y_test, nbest=2)
        res = input().strip()
        if res == "exit":
            show = False
        else:
            show = True


if __name__ == '__main__':
    main()



