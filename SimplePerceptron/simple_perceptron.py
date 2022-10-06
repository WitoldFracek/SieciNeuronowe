import numpy as np
import random


class SimplePerceptron:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, activation_function, gradient, output_mapping=lambda x: x):
        self.__weights = np.zeros(1)
        self.__x_train = x_train
        self.__y_train = y_train
        self.__init_weights(x_train.shape[0])
        self.__act_function = activation_function
        self.__gradient = gradient
        self.__mapping = np.vectorize(output_mapping)
        self.__is_trained = False
        self.__accuracy = 0

    def __init_weights(self, input_size: int):
        self.__weights = np.random.randn(1, input_size) * 0.001

    @staticmethod
    def __cost(w, x):
        return w.dot(x)

    def __activation(self, cost):
        return self.__act_function(cost)

    def __get_accuracy(self, predictions) -> float:
        transformed = self.__mapping(predictions)
        total = self.__y_train.shape[1]
        counter = 0
        for pred, exp in zip(transformed[0], self.__y_train[0]):
            if pred == exp:
                counter += 1
        return counter / total

    def train_model(self, iterations, learning_rate: float):
        activation = np.zeros(self.__y_train.shape[1])
        for _ in range(iterations):
            current_cost = self.__cost(self.__weights, self.__x_train)
            activation = self.__activation(current_cost)
            delta = self.__gradient(activation, self.__y_train, self.__x_train)
            self.__weights = self.__weights + learning_rate * delta
        self.__accuracy = self.__get_accuracy(activation)
        self.__is_trained = True

    def check_sample(self, x1: float, x2: float):
        if not self.__is_trained:
            raise Exception("Model was not trained. Please train the model before using it.")
        sample = np.array([1, x1, x2]).T
        current_cost = self.__cost(self.__weights, sample)
        activation = self.__activation(current_cost)
        return activation





