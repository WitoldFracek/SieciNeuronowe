import math
import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    return (values >= 0).astype(int) * values


def relu_der(values: np.ndarray) -> np.ndarray:
    return (values >= 0).astype(int)


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

