import math
import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    return (values >= 0).astype(int) * values


def relu_der(values: np.ndarray) -> np.ndarray:
    return (values >= 0).astype(int)


def relu_cutoff(threshold):
    def inner(values: np.ndarray) -> np.ndarray:
        gt = (values >= threshold).astype(int) * threshold
        lt = np.logical_and(values > 0, values < threshold).astype(int) * values
        return gt + lt
    return inner


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(values * (-1)))


def sigmoid_der(values: np.ndarray) -> np.ndarray:
    sig = sigmoid(values)
    return sig * (1 - sig)


def tanh(values: np.ndarray) -> np.ndarray:
    return np.tanh(values)


def tanh_der(value: np.ndarray) -> np.ndarray:
    return 1 - tanh(value) ** 2


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(values) / np.sum(np.exp(values), axis=0)


def load_mnist_data():
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.T.reshape(28 * 28, -1) / 255
    x_test = x_test.T.reshape(28 * 28, -1) / 255

    temp = []
    for elem in y_train:
        col = transform_row(elem)
        temp.append(col)
    new_y_train = np.column_stack(temp)

    temp = []
    for elem in y_test:
        col = transform_row(elem)
        temp.append(col)
    new_y_test = np.column_stack(temp)

    return x_train, new_y_train, x_test, new_y_test


def transform_row(index):
    ret = np.zeros((10, 1))
    ret[index] = 1
    return ret
