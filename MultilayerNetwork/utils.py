import math
import numpy as np


@np.vectorize
def relu(value):
    return max(0, value)


@np.vectorize
def relu_der(value):
    return 1 if value > 0 else 0


@np.vectorize
def sigmoid(value):
    e_value = math.exp(-value)
    return 1 / (1 + e_value)


@np.vectorize
def sigmoid_der(value):
    e_value = math.exp(-value)
    sig = 1 / (1 + e_value)
    return sig * (1 - sig)


@np.vectorize
def tanh(value):
    e_value = math.exp(-2 * value)
    return 2 / (1 + e_value) - 1


@np.vectorize
def tanh_der(value):
    # e_value = math.exp(-2 * value)
    # th = 2 / (1 + e_value) - 1
    # return 1 - th ** 2
    return 1 - tanh(value) ** 2


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(values) / np.sum(np.exp(values), axis=0)


def softmax_column(values):
    return np.exp(values) / np.sum(np.exp(values))
