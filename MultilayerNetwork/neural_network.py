from layer import Layer
import numpy as np


class NeuralNetwork:
    def __init__(self, layers_sizes: list[tuple[int, int]],
                 act_functions: list,
                 act_derivatives: list,
                 init_weights_range=None,
                 learning_rate=0.01):
        self.__layers = []
        init_weights = [init_weights_range] * len(layers_sizes) if init_weights_range else [None] * len(layers_sizes)
        for i in range(len(layers_sizes)):
            layer = Layer(size=layers_sizes[i],
                          activation=act_functions[i],
                          act_derivative=act_derivatives[i],
                          init_weights_range=init_weights[i],
                          learning_rate=learning_rate)
            self.__layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in self.__layers:
            a = layer.forward(a)
        return a

    def backward(self, y: np.ndarray):
        last_layer = self.__layers[-1]
        dz = last_layer.last_layer_operations(y)
        w = last_layer.weights
        for layer in self.__layers[-2::-1]:
            w, dz = layer.backward(w, dz)

    def update(self):
        for layer in self.__layers:
            layer.update()






