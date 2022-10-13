import numpy as np
import random

import numpy.random


class Adaline:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, input_size: int, output_size: int,
                 output_mapping=lambda x: x):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__set_size = x_train.shape[1]
        self.__weights = self.__init_weights(input_size, output_size)
        self.__output_map = np.vectorize(output_mapping)
        self.__train_accuracy = 0
        self.__test_accuracy = 0
        self.__training_iterations = 0

    def __init_weights(self, input_size, output_size):
        return np.random.randn(output_size, input_size) * 0.01

    def __cost(self, weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        return weights.dot(x)

    def __square_err(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return np.mean(np.square(expected - self.__output_map(predicted)))

    def __err(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return expected - self.__output_map(predicted)

    def __get_accuracy(self, expected: np.ndarray, predicted: np.ndarray) -> float:
        transformed = self.__output_map(predicted)
        mean = np.mean(expected[0] == transformed[0])
        return float(mean)

    def train(self, min_error: float, learning_rate: float):
        cost = np.zeros(self.__y_train.shape[1])
        sqr_err = 1e10
        while sqr_err > min_error:
            self.__training_iterations += 1
            cost = self.__cost(self.__weights, self.__x_train)
            error = self.__err(self.__y_train, cost)
            sqr_err = self.__square_err(self.__y_train, cost)
            dw = error.dot(self.__x_train.T)
            self.__weights = self.__weights + learning_rate * dw
        self.__train_accuracy = self.__get_accuracy(self.__y_train, cost)

    def test(self, x_test, y_test):
        cost = self.__cost(self.__weights, x_test)
        self.__test_accuracy = self.__get_accuracy(y_test, cost)

    @property
    def train_accuracy(self):
        return self.__train_accuracy

    @property
    def weights(self):
        return self.__weights

    @property
    def test_accuracy(self):
        return self.__test_accuracy

    @property
    def iterations(self):
        return self.__training_iterations


