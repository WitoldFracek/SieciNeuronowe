import color
from perceptron_utils import *
from simple_perceptron import SimplePerceptron
from adaline_perceptron import Adaline
import numpy as np

SET_SIZE = 100
TRAIN_PERCENTAGE = 0.8
ALPHA = 0.2
ACT_FUN = ActFun.UNIPOLAR
WEIGHT_RANGE = 0.1
INPUT_DATA = OR_VALUES

SHOW_PLOT = True

MIN_ERR = 0.1

ACT_FUNCTIONS = {
    ActFun.UNIPOLAR: (unipolar_activation, lambda x: x, lambda x: x),
    ActFun.BIPOLAR: (bipolar_activation, lambda x: x if round(x) == 1 else -1 + x, lambda x: -1 if x == 0 else 1)
}

train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
X_SET, Y_SET = generate_set(SET_SIZE, OR_VALUES)
X_TRAIN, Y_TRAIN = X_SET[..., :train_last_index], Y_SET[..., :train_last_index]
X_TEST, Y_TEST = X_SET[..., train_last_index:], Y_SET[..., train_last_index:]


def perceptron_theta(activation_set):
    """
    Starts with small weights and some sample values of theta.
    """
    activation, in_map, out_map = activation_set
    theta_values = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    print(f"theta\tmean acc\tdeviation acc\tmean steps\tdeviation steps")
    for theta in theta_values:
        accuracy = []
        steps = []
        for _ in range(10):
            perceptron = SimplePerceptron(X_TRAIN, Y_TRAIN,
                                          activation_function=activation,
                                          activation_theta=theta,
                                          input_mapping=in_map,
                                          output_mapping=out_map,
                                          weight_range=0.01
                                          )
            perceptron.train_model(ALPHA)
            steps.append(perceptron.iterations)
            perceptron.test_model(X_TEST, Y_TEST)
            accuracy.append(perceptron.test_accuracy)
        mean_accuracy = float(np.mean(np.asarray(accuracy)))
        mean_steps = float(np.mean(np.asarray(steps)))
        dev_accuracy = float(np.std(np.asarray(accuracy)))
        dev_steps = float(np.std(np.asarray(steps)))
        print(f" {theta:.1f}\t{mean_accuracy:.3f}\t{dev_accuracy:.3f}\t{mean_steps:.3f}\t{dev_steps:.3f}")


def uni_vs_bi_alfa():
    activation, in_map, out_map = ACT_FUNCTIONS[ActFun.UNIPOLAR]
    print(f"alfa\tmean acc\tdeviation acc\tmean steps\tdeviation steps")
    accuracy = []
    steps = []
    perceptron = SimplePerceptron(X_TRAIN, Y_TRAIN,
                                  activation_function=activation,
                                  activation_theta=0.0,
                                  input_mapping=in_map,
                                  output_mapping=out_map,
                                  weight_range=0.01
                                  )
    for _ in range(10):
        perceptron.train_model(0.4)
        steps.append(perceptron.iterations)
        perceptron.test_model(X_TEST, Y_TEST)
        accuracy.append(perceptron.test_accuracy)
    mean_accuracy = float(np.mean(np.asarray(accuracy)))
    mean_steps = float(np.mean(np.asarray(steps)))
    dev_accuracy = float(np.std(np.asarray(accuracy)))
    dev_steps = float(np.std(np.asarray(steps)))
    print(f"{0.4:.2f}\t{mean_steps:.3f}\t{dev_steps:.3f}\t{mean_accuracy:.3f}\t{dev_accuracy:.3f}".replace('.', ','))

    activation, in_map, out_map = ACT_FUNCTIONS[ActFun.BIPOLAR]
    print(f"alfa\tmean acc\tdeviation acc\tmean steps\tdeviation steps")
    accuracy = []
    steps = []
    perceptron = SimplePerceptron(X_TRAIN, Y_TRAIN,
                                  activation_function=activation,
                                  activation_theta=0.0,
                                  input_mapping=in_map,
                                  output_mapping=out_map,
                                  weight_range=0.01
                                  )
    for _ in range(10):
        perceptron.train_model(0.2)
        steps.append(perceptron.iterations)
        perceptron.test_model(X_TEST, Y_TEST)
        accuracy.append(perceptron.test_accuracy)
    mean_accuracy = float(np.mean(np.asarray(accuracy)))
    mean_steps = float(np.mean(np.asarray(steps)))
    dev_accuracy = float(np.std(np.asarray(accuracy)))
    dev_steps = float(np.std(np.asarray(steps)))
    print(f"{0.2:.2f}\t{mean_steps:.3f}\t{dev_steps:.3f}\t{mean_accuracy:.3f}\t{dev_accuracy:.3f}".replace('.', ','))


def test_alfa_on_training(activation_set):
    activation, in_map, out_map = activation_set
    alpha_values = [x/100 for x in range(101)]
    for alpha in alpha_values:
        steps = []
        accuracy = []
        for _ in range(100):
            perceptron = SimplePerceptron(X_TRAIN, Y_TRAIN,
                                          activation_function=activation,
                                          activation_theta=0.0,
                                          input_mapping=in_map,
                                          output_mapping=out_map,
                                          weight_range=0.01
                                          )
            perceptron.train_model(0.2)
            steps.append(perceptron.iterations)
            perceptron.test_model(X_TEST, Y_TEST)
            accuracy.append(perceptron.test_accuracy)
        mean_accuracy = float(np.mean(np.asarray(accuracy)))
        mean_steps = float(np.mean(np.asarray(steps)))
        dev_accuracy = float(np.std(np.asarray(accuracy)))
        dev_steps = float(np.std(np.asarray(steps)))
        print(f"{alpha:.2f}\t{mean_steps:.3f}\t{dev_steps:.3f}\t{mean_accuracy:.3f}\t{dev_accuracy:.3f}".replace('.', ','))


def test_perceptron_weights_range(activation_set):
    activation, in_map, out_map = activation_set
    weights_range = [0.1, 0.3, 0.5, 0.8, 1]
    for weight_range in weights_range:
        steps = []
        accuracy = []
        for _ in range(100):
            perceptron = SimplePerceptron(X_TRAIN, Y_TRAIN,
                                          activation_function=activation,
                                          activation_theta=0.0,
                                          input_mapping=in_map,
                                          output_mapping=out_map,
                                          weight_range=weight_range
                                          )
            perceptron.train_model(0.2)
            steps.append(perceptron.iterations)
            perceptron.test_model(X_TEST, Y_TEST)
            accuracy.append(perceptron.test_accuracy)
        mean_accuracy = float(np.mean(np.asarray(accuracy)))
        mean_steps = float(np.mean(np.asarray(steps)))
        dev_accuracy = float(np.std(np.asarray(accuracy)))
        dev_steps = float(np.std(np.asarray(steps)))
        print(f"{weight_range:.2f}\t{mean_steps:.3f}\t{dev_steps:.3f}\t{mean_accuracy:.3f}\t{dev_accuracy:.3f}".replace('.', ','))


def test_adaline_mi_parameter():
    mi_values = [x / 10000 for x in range(175)]
    for mi in mi_values:
        steps = []
        accuracy = []
        for _ in range(100):
            adaline = Adaline(X_TRAIN, Y_TRAIN,
                              input_size=3,
                              output_size=1,
                              output_mapping=lambda x: 1 if x > 0 else -1,
                              init_weight_range=WEIGHT_RANGE)
            adaline.train(MIN_ERR, mi)
            steps.append(adaline.iterations)
            adaline.test(X_TEST, Y_TEST)
            accuracy.append(adaline.test_accuracy)
        mean_accuracy = float(np.mean(np.asarray(accuracy)))
        mean_steps = float(np.mean(np.asarray(steps)))
        dev_accuracy = float(np.std(np.asarray(accuracy)))
        dev_steps = float(np.std(np.asarray(steps)))
        print(f"{mi:.4f}\t{mean_steps:.3f}\t{dev_steps:.3f}\t{mean_accuracy:.3f}\t{dev_accuracy:.3f}".replace('.', ','))


if __name__ == '__main__':
    pass
    # perceptron_theta(ACT_FUNCTIONS[ActFun.UNIPOLAR])
    # perceptron_theta(ACT_FUNCTIONS[ActFun.BIPOLAR])
    # test_alfa_on_training(ACT_FUNCTIONS[ActFun.BIPOLAR])
    # test_perceptron_weights_range(ACT_FUNCTIONS[ActFun.UNIPOLAR])
    # test_perceptron_weights_range(ACT_FUNCTIONS[ActFun.BIPOLAR])
    test_adaline_mi_parameter()


