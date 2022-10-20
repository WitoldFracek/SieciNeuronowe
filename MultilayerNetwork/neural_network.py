from layer import Layer


class NeuralNetwork:
    def __init__(self, layers_sizes: list[tuple[int, int]],
                 act_functions: list,
                 act_derivatives: list,
                 init_weights_range=None,
                 learning_rate=0.01):
        self.__layers = []
        init_weights = init_weights_range if init_weights_range else [None] * len(layers_sizes)
        for i in range(len(layers_sizes)):
            layer = Layer(layers_sizes[i],
                          act_functions[i],
                          act_derivatives[i],
                          init_weights[i],
                          learning_rate)
            self.__layers.append(layer)




