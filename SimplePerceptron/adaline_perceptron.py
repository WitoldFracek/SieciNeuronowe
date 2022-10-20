import numpy as np
import random

import numpy.random

import color


class Adaline:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, input_size: int, output_size: int,
                 output_mapping=lambda x: x, init_weight_range=1):
        self.__x_train = x_train
        self.__y_train = np.vectorize(output_mapping)(y_train)
        self.__set_size = x_train.shape[1]
        self.__weights = self.__init_weights(input_size, output_size, init_weight_range)
        self.__output_map = np.vectorize(output_mapping)
        self.__train_accuracy = 0
        self.__test_accuracy = 0
        self.__training_iterations = 0
        self.__idle_error = 0
        self.__last_error = 1e10

    def __init_weights(self, input_size, output_size, weights_range):
        return (np.random.random((output_size, input_size)) - 0.5) * weights_range

    def __cost(self, weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        return weights.dot(x)

    def __square_err(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return np.mean(np.square(expected - predicted))

    def __err(self, expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return expected - predicted

    def __get_accuracy(self, expected: np.ndarray, predicted: np.ndarray) -> float:
        transformed = np.sign(predicted)
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
            if sqr_err < self.__last_error:
                self.__last_error = sqr_err
                self.__idle_error = 0
            else:
                self.__idle_error += 1
            if self.__idle_error > 10:
                # print(f"{color.Color.FG.RED}Break! Too many iterations without error change.{color.Color.END}")
                # print(f"{color.Color.FG.RED}Current minimal error: {sqr_err:.3f}{color.Color.END}")
                # print(f"{color.Color.FG.RED}Given minimal error:   {min_error:.3f}{color.Color.END}")
                # print(f"{color.Color.FG.RED}{sqr_err:.3f} > {min_error:.3f}{color.Color.END}")
                break
            dw = error.dot(self.__x_train.T)
            self.__weights = self.__weights + learning_rate * dw
        self.__train_accuracy = self.__get_accuracy(self.__y_train, cost)

    def test(self, x_test, y_test):
        cost = self.__cost(self.__weights, x_test)
        self.__test_accuracy = self.__get_accuracy(self.__output_map(y_test), cost)

    def test_one_sample(self, x1, x2):
        current_cost = self.__cost(self.__weights, np.array([1, x1, x2]).reshape(3, 1))
        return 1 if self.__output_map(current_cost)[0][0] == 1 else 0

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


