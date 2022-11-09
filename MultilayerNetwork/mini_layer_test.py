from neural_network import NeuralNetwork
from functions import *
import numpy as np
from binary_utils import *
from keras.datasets import mnist
import tensorflow
from tqdm import tqdm
import matplotlib.pyplot as pyl
from data_transform import make_batch, load_mnist_data, shuffle_data, show_random, transform_row_to_one_hit, false_values_generator, show_bad
from experiments import make_neural_network
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.set_printoptions(precision=3, suppress=True)

LAYER_SIZES = [(28 * 28, 50), (50, 40), (40, 30), (30, 10)]
ACT_FUNCTIONS = [tanh] * len(LAYER_SIZES)
ACT_FUNCTIONS[-1] = softmax
ACT_DERIVATIVES = [tanh_der] * len(LAYER_SIZES)
LEARNING_RATE = 0.5
ITERATIONS = 1
BATCH_SIZE = 200
BATCH_NUMBER = 100
STANDARD_DEV = 1
MEAN = 0


def main():
    nn = NeuralNetwork(LAYER_SIZES,
                       ACT_FUNCTIONS,
                       ACT_DERIVATIVES,
                       learning_rate=LEARNING_RATE,
                       standard_dev=STANDARD_DEV,
                       mean=MEAN)
    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle_data(x_train, y_train)

    training_sets = [make_batch(BATCH_SIZE, x_train, y_train) for _ in range(BATCH_NUMBER)]
    for _ in tqdm(range(ITERATIONS), colour='GREEN'):
        for x_batch, y_batch in training_sets:
            nn.forward(x_batch)
            nn.backward(y_batch)
            nn.update()

    predictions = nn.forward(x_test)
    y_pred = np.argmax(predictions, axis=0)
    y_true = np.argmax(y_test, axis=0)
    print(np.mean(y_pred == y_true))

    for weight in nn.weights:
        pyl.imshow(weights_to_image(weight))
        pyl.show()

    while True:
        a = get_random_input(x_test)
        pred = nn.forward(a)
        y_pred = np.argmax(pred)
        print(y_pred)
        for weight in nn.weights:
            z = weight.dot(a) + 1
            pyl.imshow(z)
            pyl.show()
            a = tanh(z)
        res = input()
        if res == 'exit':
            break


def weights_to_image(weights: np.ndarray) -> np.ndarray:
    mn = np.min(weights)
    ret = weights - mn
    mx = np.max(ret)
    return (ret / mx * 255).astype(np.uint8)


def multiply_weights(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights.dot(x)
    return weights_to_image(w)


def get_random_input(x: np.ndarray):
    index = random.randint(0, x.shape[1])
    return x[:, index].reshape(-1, 1)


if __name__ == '__main__':
    main()
