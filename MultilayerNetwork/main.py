from neural_network import NeuralNetwork
from functions import *
import numpy as np
from binary_utils import *
from keras.datasets import mnist
import tensorflow
from tqdm import tqdm
import matplotlib.pyplot as pyl
from data_transform import make_batch, load_mnist_data, shuffle_data, show_random, transform_row_to_one_hit, false_values_generator, show_bad
from experiments import make_neural_network
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.set_printoptions(precision=3, suppress=True)

# LAYER_SIZES = [(28 * 28, 70), (70, 50), (50, 10)]  # accuracy 91%
# LAYER_SIZES = [(28 * 28, 50), (50, 40), (40, 30), (30, 10)]
LAYER_SIZES = [(28 * 28, 50), (50, 10)]
ACT_FUNCTIONS = [tanh] * len(LAYER_SIZES)
ACT_FUNCTIONS[-1] = softmax
ACT_DERIVATIVES = [tanh_der] * len(LAYER_SIZES)
LEARNING_RATE = 0.5
ITERATIONS = 200
BATCH_SIZE = 200
BATCH_NUMBER = 100
STANDARD_DEV = 1
MEAN = 0


def get_neurons_in_layers(layers):
    return ", ".join([str(n) for (p, n) in layers])


def get_accuracy(pred, label):
    y_pred = np.argmax(pred, axis=0)
    y_true = np.argmax(label, axis=0)
    return np.mean(y_pred == y_true)


def test_act_fun():
    layers = [(784, 50), (50, 10)]
    learning_rate = 0.5
    std_dev = 0.01
    iterations = 100
    batch_size = 500
    batch_count = 300

    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle_data(x_train, y_train)

    training_sets = [make_batch(batch_size, x_train, y_train) for _ in range(batch_count)]

    relu_nn = make_neural_network(layers, relu, relu_der, learning_rate, std_dev, 0)
    sigmoid_nn = make_neural_network(layers, sigmoid, sigmoid_der, learning_rate, std_dev, 0)
    tanh_nn = make_neural_network(layers, tanh, tanh_der, learning_rate, std_dev, 0)

    relu_err = []
    sigmoid_err = []
    tanh_err = []

    for _ in tqdm(range(iterations)):
        for x_batch, y_batch in training_sets:
            relu_nn.forward(x_batch); sigmoid_nn.forward(x_batch); tanh_nn.forward(x_batch)
            relu_nn.backward(y_batch); sigmoid_nn.backward(y_batch); tanh_nn.backward(y_batch)
            relu_nn.update(); sigmoid_nn.update(); tanh_nn.update()
        x_exmpl, y_exmpl = training_sets[-1]
        relu_err.append(relu_nn.get_error(mean_squared_error, x_exmpl, y_exmpl))
        sigmoid_err.append(sigmoid_nn.get_error(mean_squared_error, x_exmpl, y_exmpl))
        tanh_err.append(tanh_nn.get_error(mean_squared_error, x_exmpl, y_exmpl))

    pyl.plot(range(0, len(relu_err)), relu_err, c='r', label="relu")
    pyl.plot(range(0, len(sigmoid_err)), sigmoid_err, c='b', label="sigmoid")
    pyl.plot(range(0, len(tanh_err)), tanh_err, c='g', label="tanh")
    pyl.title(f"Mean squared error\nLayers: {get_neurons_in_layers(layers)}")
    pyl.xlabel("iteration")
    pyl.ylabel("error")
    pyl.legend(loc='upper right', frameon=False)
    pyl.show()


def test_act_accuracy():
    layers = [(784, 50), (50, 10)]
    learning_rate = 0.5
    std_dev = 0.01
    iterations = 100
    batch_size = 500
    batch_count = 300

    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle_data(x_train, y_train)

    training_sets = [make_batch(batch_size, x_train, y_train) for _ in range(batch_count)]

    relu_nn = make_neural_network(layers, relu, relu_der, learning_rate, std_dev, 0)
    sigmoid_nn = make_neural_network(layers, sigmoid, sigmoid_der, learning_rate, std_dev, 0)
    tanh_nn = make_neural_network(layers, tanh, tanh_der, learning_rate, std_dev, 0)

    relu_acc = []
    sigmoid_acc = []
    tanh_acc = []

    for _ in tqdm(range(iterations)):
        for x_batch, y_batch in training_sets:
            relu_nn.forward(x_batch)
            sigmoid_nn.forward(x_batch)
            tanh_nn.forward(x_batch)
            relu_nn.backward(y_batch)
            sigmoid_nn.backward(y_batch)
            tanh_nn.backward(y_batch)
            relu_nn.update()
            sigmoid_nn.update()
            tanh_nn.update()
        relu_acc.append(get_accuracy(relu_nn.forward(x_test), y_test))
        sigmoid_acc.append(get_accuracy(sigmoid_nn.forward(x_test), y_test))
        tanh_acc.append(get_accuracy(tanh_nn.forward(x_test), y_test))

    pyl.plot(range(0, len(relu_acc)), relu_acc, c='r', label="relu")
    pyl.plot(range(0, len(sigmoid_acc)), sigmoid_acc, c='b', label="sigmoid")
    pyl.plot(range(0, len(tanh_acc)), tanh_acc, c='g', label="tanh")
    pyl.title(f"Accuracy\nLayers: {get_neurons_in_layers(layers)}")
    pyl.xlabel("iteration")
    pyl.ylabel("accuracy")
    pyl.legend(loc='lower right', frameon=False)
    pyl.show()


def test_layer_size():
    learning_rate = 0.5
    std_dev = 0.1
    iterations = 100
    batch_size = 500
    batch_count = 300

    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle_data(x_train, y_train)

    training_sets = [make_batch(batch_size, x_train, y_train) for _ in range(batch_count)]

    small = make_neural_network([(784, 20), (20, 10), (10, 10)], tanh, tanh_der, learning_rate, std_dev, 0)
    medium = make_neural_network([(784, 40), (40, 20), (20, 10)], tanh, tanh_der, learning_rate, std_dev, 0)
    large = make_neural_network([(784, 100), (100, 50), (50, 10)], tanh, tanh_der, learning_rate, std_dev, 0)

    small_acc = []
    small_err = []
    medium_acc = []
    medium_err = []
    large_acc = []
    large_err = []

    for _ in tqdm(range(iterations)):
        for x_batch, y_batch in training_sets:
            small.forward(x_batch)
            medium.forward(x_batch)
            large.forward(x_batch)
            small.backward(y_batch)
            medium.backward(y_batch)
            large.backward(y_batch)
            small.update()
            medium.update()
            large.update()
        x_exmpl, y_exmpl = training_sets[-1]
        small_err.append(small.get_error(mean_squared_error, x_exmpl, y_exmpl))
        medium_err.append(medium.get_error(mean_squared_error, x_exmpl, y_exmpl))
        large_err.append(large.get_error(mean_squared_error, x_exmpl, y_exmpl))
        small_acc.append(get_accuracy(small.forward(x_test), y_test))
        medium_acc.append(get_accuracy(medium.forward(x_test), y_test))
        large_acc.append(get_accuracy(large.forward(x_test), y_test))

    pyl.plot(range(0, len(small_err)), small_err, c='r', label="40 neurons")
    pyl.plot(range(0, len(medium_err)), medium_err, c='b', label="70 neurons")
    pyl.plot(range(0, len(large_err)), large_err, c='g', label="160 neurons")
    pyl.title(f"Mean squared error")
    pyl.xlabel("iteration")
    pyl.ylabel("error")
    pyl.legend(loc='upper right', frameon=False)
    pyl.show()

    pyl.plot(range(0, len(small_acc)), small_acc, c='r', label="40 neurons")
    pyl.plot(range(0, len(medium_acc)), medium_acc, c='b', label="70 neurons")
    pyl.plot(range(0, len(large_acc)), large_acc, c='g', label="160 neurons")
    pyl.title(f"Accuracy")
    pyl.xlabel("iteration")
    pyl.ylabel("accuracy")
    pyl.legend(loc='lower right', frameon=False)
    pyl.show()


def get_confusion_matrix(y_true, y_pred, normalize='true'):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return cm


def main():
    nn = NeuralNetwork(LAYER_SIZES,
                       ACT_FUNCTIONS,
                       ACT_DERIVATIVES,
                       learning_rate=LEARNING_RATE,
                       standard_dev=STANDARD_DEV,
                       mean=MEAN)

    # x_train, y_train = generate_set(1000, XOR_VALUES, operator_xor)
    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle_data(x_train, y_train)

    training_sets = [make_batch(BATCH_SIZE, x_train, y_train) for _ in range(BATCH_NUMBER)]
    errs = []
    for _ in tqdm(range(ITERATIONS), colour='GREEN'):
        for x_batch, y_batch in training_sets:
            nn.forward(x_batch)
            nn.backward(y_batch)
            nn.update()
        x_exmpl, y_exmpl = training_sets[-1]
        err = nn.get_error(mean_squared_error, x_exmpl, y_exmpl)
        errs.append(err)

    predictions = nn.forward(x_test)

    y_pred = np.argmax(predictions, axis=0)
    y_true = np.argmax(y_test, axis=0)
    print(np.mean(y_pred == y_true))
    cm = get_confusion_matrix(y_true, y_pred, normalize=None)
    display = ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    fig, ax1 = pyl.subplots(1, 1)
    display.plot(ax=ax1, cmap='Blues')
    pyl.show()

    # for pred_col, label, x in false_values_generator(x_test, predictions, y_test):
    #     show_bad(pred_col, label, x)
    #     res = input().strip()
    #     if res == "exit":
    #         break

    show = True
    while show:
        show_random(x_test, predictions, y_test, nbest=1)
        res = input().strip()
        if res == "exit":
            show = False
        else:
            show = True


if __name__ == '__main__':
    test_layer_size()



