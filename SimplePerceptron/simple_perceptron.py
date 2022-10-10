import numpy as np
import random


class SimplePerceptron:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, activation_function, gradient, output_mapping=lambda x: x):
        self.__weights = np.zeros(1)
        self.__bias = 0 #random.random()
        self.__x_train = x_train
        self.__y_train = y_train
        self.__init_weights(x_train.shape[0])
        self.__act_function = activation_function
        self.__gradient = gradient
        self.__act_derivative = lambda x: x
        self.__mapping = np.vectorize(output_mapping)
        self.__is_trained = False
        self.__accuracy = 0
        self.__test_accuracy = 0

    def __init_weights(self, input_size: int):
        self.__weights = np.random.randn(1, input_size) * 0.001
        # self.__weights = (np.random.random((1, input_size)) * 2 - 1) * 0.001

    def __cost(self, w, x):
        return w.dot(x) + self.__bias

    def __activation(self, cost):
        return self.__act_function(cost, theta=0)

    def __error(self, predictions):
        return self.__y_train - predictions

    def __get_accuracy(self, expected, predictions) -> float:
        transformed = self.__mapping(predictions)
        mean = np.mean(expected == transformed)
        return float(mean)

    def train_model(self, iterations, learning_rate: float):
        activation = np.zeros(self.__y_train.shape[1])
        for _ in range(iterations):
            current_cost = self.__cost(self.__weights, self.__x_train)
            activation = self.__activation(current_cost)
            error = self.__error(activation)
            dw = error.dot(self.__x_train.T)
            # db = np.sum(self.__act_derivative(error)) / self.__x_train.shape[1]
            self.__weights = self.__weights + learning_rate * dw
            # self.__bias = self.__bias + learning_rate * db
        self.__accuracy = self.__get_accuracy(self.__y_train, activation)
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
    def weights(self):
        return self.__weights

    @property
    def bias(self):
        return self.__bias




