import color
from simple_perceptron import SimplePerceptron
from perceptron_utils import *

SET_SIZE = 100
TRAIN_PERCENTAGE = 0.8
ALPHA = 0.2
ACT_FUN = ActFun.UNIPOLAR
WEIGHT_RANGE = 0.1
SHOW_PLOT = True


ACT_FUNCTIONS = {
    ActFun.UNIPOLAR: (unipolar_activation, lambda x: x, lambda x: x),
    ActFun.BIPOLAR: (bipolar_activation, lambda x: x if round(x) == 1 else -1 + x, lambda x: -1 if x == 0 else 1)
}


def main():
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, AND_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    act_fun, input_mapping, output_mapping = ACT_FUNCTIONS[ACT_FUN]
    perceptron = SimplePerceptron(x_train, y_train, act_fun, input_mapping=input_mapping, output_mapping=output_mapping)
    perceptron.train_model(ALPHA)

    weights = perceptron.weights[0] / perceptron.weights[0][0]
    print(f"\nModel: {color.Color.FG.VIOLET}{perceptron.__class__.__name__.upper()}{color.Color.END}")
    print(f"Initial set size: {color.Color.FG.VIOLET}{SET_SIZE}{color.Color.END} | Training size: {color.Color.FG.VIOLET}{train_last_index}{color.Color.END}")
    print(f"Learning rate: {color.Color.FG.VIOLET}{ALPHA}{color.Color.END}")
    print(f"Weight range: {color.Color.FG.VIOLET}{WEIGHT_RANGE}{color.Color.END}")
    print(f"Activation: {color.Color.FG.VIOLET}{ACT_FUN}{color.Color.END}")
    print(f"\nTrain accuracy: - {perceptron.train_accuracy}")
    print(f"Iterations: ----- {perceptron.iterations}")
    print(f"Weights: -------- {weights[0]:.2f} | {weights[1]:.2f} | {weights[2]:.2f}")
    perceptron.test_model(x_test, y_test)
    print(f"\nTest accuracy: -- {perceptron.train_accuracy}")
    if SHOW_PLOT:
        plot_result(np.vectorize(input_mapping)(x_train),
                    np.vectorize(output_mapping)(y_train),
                    weights,
                    plot_range=(-1.25, 1.25) if ACT_FUN == ActFun.BIPOLAR else (-0.25, 1.25)
                    )


if __name__ == '__main__':
    main()
