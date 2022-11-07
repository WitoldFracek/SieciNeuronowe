from neural_network import NeuralNetwork
from functions import *
import numpy as np
from binary_utils import *
from keras.datasets import mnist
import tensorflow
from tqdm import tqdm
import matplotlib.pyplot as pyl
from data_transform import make_batch, load_mnist_data, shuffle_data, show_random, transform_row_to_one_hit

np.set_printoptions(precision=3, suppress=True)

# LAYER_SIZES = [(28 * 28, 70), (70, 50), (50, 10)]  # accuracy 91%
LAYER_SIZES = [(28 * 28, 50), (50, 40), (40, 30), (30, 10)]
# LAYER_SIZES = [(28 * 28, 50), (50, 10)]
ACT_FUNCTIONS = [sigmoid] * len(LAYER_SIZES)
ACT_FUNCTIONS[-1] = softmax
ACT_DERIVATIVES = [sigmoid_der] * len(LAYER_SIZES)
LEARNING_RATE = 0.5
ITERATIONS = 100
BATCH_SIZE = 500
BATCH_NUMBER = 300
STANDARD_DEV = 1
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
    errs = []
    for _ in tqdm(range(ITERATIONS), colour='GREEN'):
        for x_batch, y_batch in training_sets:
            nn.forward(x_batch)
            nn.backward(y_batch)
            nn.update()
        x_exmpl, y_exmpl = training_sets[-1]
        err = nn.get_error(mean_squared_error, x_exmpl, y_exmpl)
        errs.append(err)

    predictions = nn.forward(x_test)

    values = np.argmax(predictions, axis=0)
    print(np.mean(values == np.argmax(y_test, axis=0)))

    pyl.plot(range(0, len(errs)), errs)
    pyl.show()

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



