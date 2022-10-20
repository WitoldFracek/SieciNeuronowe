import numpy as np

import color


class SimplePerceptron:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, activation_function, activation_theta=0.0,
                 input_mapping=lambda x: x, output_mapping=lambda x: x, weight_range=1):
        self.__weights = np.zeros(1)
        self.__theta = activation_theta
        self.__x_train = np.vectorize(input_mapping)(x_train)
        self.__y_train = np.vectorize(output_mapping)(y_train)
        self.__init_weights(x_train.shape[0], weight_range)
        self.__act_function = activation_function
        self.__mapping = np.vectorize(input_mapping)
        self.__output_mapping = np.vectorize(output_mapping)
        self.__is_trained = False
        self.__accuracy = 0
        self.__iterations = 0
        self.__test_accuracy = 0
        self.__last_error = 1e10
        self.__idle_error = 0

    def __init_weights(self, input_size: int, weight_range):
        self.__weights = (np.random.random((1, input_size)) - 0.5) * weight_range

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
                print(f"{color.Color.FG.RED}Break! Too many iterations without error change.{color.Color.END}")
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

    def test_one_sample(self, x1, x2):
        if not self.__is_trained:
            raise Exception("Model was not trained. Please train the model before using it.")
        current_cost = self.__cost(self.__weights, self.__mapping(np.array([1, x1, x2]).reshape(3, 1)))
        activation = self.__activation(current_cost)
        return 1 if activation[0][0] == 1 else 0

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




