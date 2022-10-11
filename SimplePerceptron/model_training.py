from simple_perceptron import SimplePerceptron
from perceptron_utils import *
import matplotlib.pyplot as pyl

SET_SIZE = 100
TRAIN_PERCENTAGE = 0.8
ALPHA = 0.2
ACT_FUN = ActFun.UNIPOLAR


ACT_FUNCTIONS = {
    ActFun.UNIPOLAR: (unipolar_activation, lambda x: x),
    ActFun.BIPOLAR: (bipolar_activation, lambda x: -1 if x == 0 else 1)
}


def main():
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, AND_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    act_fun, mapping = ACT_FUNCTIONS[ACT_FUN]
    perceptron = SimplePerceptron(x_train, y_train, act_fun, input_mapping=mapping)
    perceptron.train_model(ALPHA)

    weights = perceptron.weights[0] / perceptron.weights[0][0]
    print(f"\nInitial set size: {SET_SIZE} | Training size {train_last_index}")
    print(f"Activation: ----- {ACT_FUN}")
    print(f"Train accuracy: - {perceptron.train_accuracy}")
    print(f"Iterations: ----- {perceptron.iterations}")
    print(f"Weights: -------- {weights[0]:.2f} | {weights[1]:.2f} | {weights[2]:.2f}")
    perceptron.test_model(x_test, y_test)
    print(f"Test accuracy: -- {perceptron.train_accuracy}")
    for i in range(x_train.shape[1]):
        if y_train[0][i] == 1:
            pyl.scatter([x_train[2][i]], [x_train[1][i]], c='b')
        else:
            pyl.scatter([x_train[2][i]], [x_train[1][i]], c='r')
    x_all = np.linspace(-0.25, 1.25, 10)
    pyl.plot(x_all, (-weights[1] * x_all - weights[0]) / weights[2])
    pyl.show()


if __name__ == '__main__':
    main()
