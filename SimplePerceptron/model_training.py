from simple_perceptron import SimplePerceptron
from perceptron_utils import *

TRAIN_SIZE = 1000
ITERATIONS = 1000
ALPHA = 0.5


def main():
    x_train, y_train = generate_training_set(TRAIN_SIZE, OR_VALUES)
    perceptron = SimplePerceptron(x_train, y_train, unipolar_activation, step_function_delta)
    perceptron.train_model(ITERATIONS, ALPHA)
    res = perceptron.check_sample(-1, 1)
    print(res)


if __name__ == '__main__':
    main()



