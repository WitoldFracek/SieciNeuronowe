from simple_perceptron import SimplePerceptron
from perceptron_utils import *
import matplotlib.pyplot as pyl

SET_SIZE = 1000
TRAIN_PERCENTAGE = 0.8
ITERATIONS = 1_000
ALPHA = 0.2


def main():
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, XOR_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    pyl.scatter(x_train[0], x_train[1])
    pyl.show()
    perceptron = SimplePerceptron(x_train, y_train, bipolar_activation, step_function_delta, output_mapping=lambda x: 0 if x == -1 else 1)  # output_mapping=lambda x: 1 if x == 1 else 0
    perceptron = SimplePerceptron(x_train, y_train, unipolar_activation, step_function_delta)
    perceptron.train_model(ITERATIONS, ALPHA)
    print(f"Train accuracy: {perceptron.train_accuracy}")
    print(f"Weights:  {perceptron.weights}")
    print(f"Bias: {perceptron.bias}")
    perceptron.test_model(x_test, y_test)
    print(f"Test accuracy: {perceptron.train_accuracy}")


if __name__ == '__main__':
    main()



