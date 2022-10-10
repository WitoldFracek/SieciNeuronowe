import numpy as np
import random

import numpy.random


class Adaline:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, input_size: int, output_size: int,
                 act_function=None, output_mapping=np.vectorize(lambda x: x)):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__set_size = x_train.shape[1]
        self.__weights = self.__init_weights(input_size, output_size)
        #self.__act_function = act_function
        self.__output_map = output_mapping
        self.__train_accuracy = 0

    def __init_weights(self, input_size, output_size):
        return np.random.randn(output_size, input_size) * 0.01

    def __cost(self, weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        return weights.dot(x)

    def __square_err(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return np.mean(np.square(expected - predicted))

    def __err(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return expected - predicted

    def __get_train_accuracy(self, expected: np.ndarray, predicted: np.ndarray) -> float:
        transformed = self.__output_map(predicted)
        mean = np.mean(expected[0] == transformed[0])
        return float(mean)

    def train(self, iterations: int, learning_rate: float):
        cost = np.zeros(self.__y_train.shape[1])
        for i in range(iterations):
            cost = self.__cost(self.__weights, self.__x_train)
            # print("Cost", cost)
            error = self.__err(self.__y_train, cost)
            err = self.__square_err(self.__y_train, cost)
            # print("Error matrix", error)
            # print("Mean error: ", err)
            # input()
            dw = error.dot(self.__x_train.T)
            self.__weights = self.__weights + learning_rate * dw
        self.__train_accuracy = self.__get_train_accuracy(self.__y_train, cost)

    @property
    def train_accuracy(self):
        return self.__train_accuracy

    @property
    def weights(self):
        return self.__weights


