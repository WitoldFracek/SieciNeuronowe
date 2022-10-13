import numpy as np
import random


class SimplePerceptron:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, activation_function, activation_theta=0.0,
                 input_mapping=lambda x: x):
        self.__weights = np.zeros(1)
        self.__theta = activation_theta
        self.__x_train = x_train
        self.__init_weights(x_train.shape[0])
        self.__act_function = activation_function
        self.__mapping = np.vectorize(input_mapping)
        self.__y_train = self.__mapping(y_train)
        self.__is_trained = False
        self.__accuracy = 0
        self.__iterations = 0
        self.__test_accuracy = 0
        self.__last_error = 1e10
        self.__idle_error = 0

    def __init_weights(self, input_size: int):
        self.__weights = np.random.randn(1, input_size) * 0.001
        self.__weights = np.full((1, input_size), 1)

    def __cost(self, w, x):
        return w.dot(x)

    def __activation(self, cost):
        return self.__act_function(cost, theta=self.__theta)

    def __error(self, predictions):
        return self.__y_train - predictions

    def __get_accuracy(self, expected, predictions) -> float:
        mean = np.mean(expected == predictions)
        return float(mean)

    def train_model(self, learning_rate: float):
        activation = np.zeros(self.__y_train.shape[1])
        while self.__accuracy < 1:
            current_cost = self.__cost(self.__weights, self.__x_train)
            activation = self.__activation(current_cost)
            error = self.__error(activation)
            mean_error = np.mean(error)
            if mean_error < self.__last_error:
                self.__last_error = mean_error
                self.__idle_error = 0
            else:
                self.__idle_error += 1
            if self.__idle_error > 10:
                break
            dw = error.dot(self.__x_train.T)
            self.__weights = self.__weights + learning_rate * dw
            self.__accuracy = self.__get_accuracy(self.__y_train, activation)
            self.__iterations += 1
        self.__is_trained = True

    def test_model(self, x_test: np.ndarray, y_test: np.ndarray):
        if not self.__is_trained:
            raise Exception("Model was not trained. Please train the model before using it.")
        current_cost = self.__cost(self.__weights, x_test)
        activation = self.__activation(current_cost)
        self.__test_accuracy = self.__get_accuracy(y_test, activation)

    @property
    def train_accuracy(self):
        return self.__accuracy

    @property
    def test_accuracy(self):
        if not self.__is_trained:
            raise Exception("Model was not trained. Please train the model before using it.")
        return self.__test_accuracy

    @property
    def iterations(self):
        if not self.__is_trained:
            raise Exception("Model was not trained. The number of iterations is not known")
        return self.__iterations

    @property
    def weights(self):
        return self.__weights




