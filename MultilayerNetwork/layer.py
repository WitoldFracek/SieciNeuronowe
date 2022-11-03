import numpy as np
import random


class Layer:
    def __init__(self, size: tuple[int, int], activation, act_derivative, init_weights_range=None,
                 learning_rate=0.1,
                 std_dev=1, mean=0):
        input_size, output_size = size
        self.__weights = self.__init_weights(input_size, output_size, weights_range=init_weights_range, std_dev=std_dev, mean=mean)
        self.__act_fun = activation
        self.__act_derivative = act_derivative
        self.__x_cache: np.ndarray = np.zeros((1, 1))
        self.__a_cache: np.ndarray = np.zeros((1, 1))
        self.__z_cache: np.ndarray = np.zeros((1, 1))
        self.__dw_cache: np.ndarray = np.zeros((output_size, input_size))
        self.__db_cache: float = 0
        self.__dz_cache: np.ndarray = np.zeros((1, 1))
        self.__learning_rate = learning_rate
        self.__bias = random.random()
        self.__cluster_size = 0

    def __init_weights(self, input_size: int, output_size: int, weights_range, std_dev=1, mean=0) -> np.ndarray:
        if weights_range is None:
            return np.random.normal(mean, std_dev, size=(output_size, input_size))
            # return np.random.randn(output_size, input_size)
        return (np.random.random((output_size, input_size)) - 0.5) * weights_range * 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.__x_cache = x
        self.__cluster_size = x.shape[1]
        z = self.__weights.dot(x) + self.__bias
        a = self.__act_fun(z)
        self.__z_cache = z
        self.__a_cache = a
        return a

    def backward(self, next_w, next_dz):
        dz = next_w.T.dot(next_dz) * self.__act_derivative(self.__z_cache)
        self.__compute_deltas(dz)
        return self.__weights, dz

    def update(self):
        self.__weights = self.__weights - self.__learning_rate * self.__dw_cache
        self.__bias = self.__bias - self.__learning_rate * self.__db_cache

    def last_layer_operations(self, y):
        dz = self.__a_cache - y
        self.__dz_cache = dz
        self.__compute_deltas(dz)
        return self.__dz_cache

    def __compute_deltas(self, dz: np.ndarray):
        dw = dz.dot(self.__x_cache.T) / self.__cluster_size
        db = np.sum(dz, axis=1, keepdims=True) / self.__cluster_size
        self.__dw_cache = dw
        self.__db_cache = db

    @property
    def weights(self):
        return self.__weights

    @property
    def a_cache(self):
        return self.__a_cache

    @property
    def z_cache(self):
        return self.__z_cache

    @property
    def dw_cache(self):
        return self.__dw_cache

    @property
    def db_cache(self):
        return self.__db_cache

    @property
    def dz_cache(self):
        return self.__dz_cache





