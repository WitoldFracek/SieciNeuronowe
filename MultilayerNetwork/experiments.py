import pandas as pd
from tqdm import tqdm

from data_transform import load_mnist_data, shuffle_data, make_batch
from functions import *
from neural_network import NeuralNetwork

FILE_PATH = './experiments.txt'

LAYER_SIZES = [[(784, 50), (50, 10)],
               [(784, 50), (50, 20), (20, 10)],
               [(784, 70), (70, 50), (50, 10)],
               [(784, 50), (50, 40), (40, 30), (30, 10)],
               [(784, 20), (20, 40), (40, 30), (30, 10)]]

ACT_FUNCTIONS = [(sigmoid, sigmoid_der), (tanh, tanh_der), (relu, relu_der)]
LEARNING_RATES = [0.1, 0.5, 1]
ITERATIONS = [1, 100, 600]
BATCH_SIZES = [10, 100, 500]
BATCH_COUNTS = [1, 100, 300]
STANDARD_DEVS = [0.01, 0.1, 0.5, 1, 1.5]
MEANS = [0]


def make_neural_network(layers, act_fun, act_der, learning_rate, std_dev, mean):
    act_funs = [act_fun] * len(layers)
    act_funs[-1] = softmax
    act_ders = [act_der] * len(layers)
    return NeuralNetwork(layers, act_funs, act_ders, learning_rate=learning_rate, standard_dev=std_dev, mean=mean)


def total_number_of_neurons(layers):
    count = 0
    for _, n_out in layers:
        count += n_out
    return count


def run_experiments():
    counter = 0
    all_count = len(LAYER_SIZES) * len(ACT_FUNCTIONS) * len(LEARNING_RATES) * len(ITERATIONS) * len(BATCH_SIZES) * len(BATCH_COUNTS) * len(STANDARD_DEVS)
    x_train, y_train, x_test, y_test = load_mnist_data()
    for std_dev in STANDARD_DEVS:
        for learning_rate in LEARNING_RATES:
            for iterations in ITERATIONS:
                for batch_size in BATCH_SIZES:
                    for batch_count in BATCH_COUNTS:
                        training_sets = [make_batch(batch_size, x_train, y_train) for _ in range(batch_count)]
                        for layers in LAYER_SIZES:
                            for act_fun, act_der in ACT_FUNCTIONS:
                                nn = make_neural_network(layers, act_fun, act_der, learning_rate, std_dev, 0)
                                for _ in range(iterations):
                                    for x_batch, y_batch in training_sets:
                                        nn.forward(x_batch)
                                        nn.backward(y_batch)
                                        nn.update()
                                train_err = nn.get_error(mean_squared_error, x_train, y_train)
                                test_err = nn.get_error(mean_squared_error, x_test, y_test)
                                neurons = total_number_of_neurons(layers)
                                layers_count = len(layers)

                                test_pred = nn.forward(x_test)
                                test_values = np.argmax(test_pred, axis=0)
                                test_score = float(np.mean(test_values == np.argmax(y_test, axis=0)))

                                train_pred = nn.forward(x_train)
                                train_values = np.argmax(train_pred, axis=0)
                                train_score = float(np.mean(train_values == np.argmax(y_train, axis=0)))

                                counter += 1
                                with open(FILE_PATH, 'a', encoding='utf-8') as file:
                                    line = f'\n{act_fun.__name__}\t{layers_count}\t{neurons}\t{learning_rate}' \
                                            f'\t{iterations}\t{batch_size}\t{batch_count}\t{std_dev}' \
                                            f'\t{train_err:.4f}\t{test_err:.4f}\t{train_score * 100:.2f}%' \
                                            f'\t{test_score * 100:.2f}%'.replace('.', ',')
                                    line += f'\t\t{layers}'
                                    file.write(line)
                                print(f'Complete: {counter} out of {all_count}\t{counter / all_count * 100:.2f}%')


if __name__ == '__main__':
    run_experiments()
