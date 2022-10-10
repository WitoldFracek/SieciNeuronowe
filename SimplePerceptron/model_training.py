from simple_perceptron import SimplePerceptron
from perceptron_utils import *
import matplotlib.pyplot as pyl

SET_SIZE = 1000
TRAIN_PERCENTAGE = 0.8
ITERATIONS = 1000
ALPHA = 0.2


def main():
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, AND_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    perceptron = SimplePerceptron(x_train, y_train, bipolar_activation, step_function_delta, output_mapping=lambda x: 0 if x == -1 else 1)  # output_mapping=lambda x: 1 if x == 1 else 0
    # perceptron = SimplePerceptron(x_train, y_train, unipolar_activation, step_function_delta)
    perceptron.train_model(ITERATIONS, ALPHA)
    print(f"Train accuracy: {perceptron.train_accuracy}")
    print(f"Weights:  {perceptron.weights}")
    print(f"Bias: {perceptron.bias}")
    perceptron.test_model(x_test, y_test)
    weights = perceptron.weights
    for i in range(x_train.shape[1]):
        if y_train[0][i] == 1:
            pyl.scatter([x_train[2][i]], [x_train[1][i]], c='b')
        else:
            pyl.scatter([x_train[2][i]], [x_train[1][i]], c='r')
    x_all = np.linspace(-0.25, 1.25, 10)
    pyl.plot(x_all, (-weights[0][1] * x_all - weights[0][0]) / weights[0][2])
    pyl.show()
    print(f"Test accuracy: {perceptron.train_accuracy}")


if __name__ == '__main__':
    main()
