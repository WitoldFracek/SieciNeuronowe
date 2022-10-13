import color
from adaline_perceptron import Adaline
from perceptron_utils import AND_VALUES, OR_VALUES, XOR_VALUES, generate_training_set, plot_result

SET_SIZE = 1000
TRAIN_PERCENTAGE = 0.8
MIN_ERR = 0.25
ALPHA = 0.001
WEIGHT_RANGE = 0.03
SHOW_PLOT = False


def main():
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, OR_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    adaline = Adaline(x_train, y_train, input_size=3, output_size=1, output_mapping=lambda x: 1 if x > 0 else -1)
    adaline.train(MIN_ERR, ALPHA)
    weights = adaline.weights[0] / adaline.weights[0][0]
    print(f"\nModel: {color.Color.FG.VIOLET}{adaline.__class__.__name__.upper()}{color.Color.END}")
    print(f"Initial set size: {color.Color.FG.VIOLET}{SET_SIZE}{color.Color.END} | Training size: {color.Color.FG.VIOLET}{train_last_index}{color.Color.END}")
    print(f"Learning rate: {color.Color.FG.VIOLET}{ALPHA}{color.Color.END}")
    print(f"Weight range: {color.Color.FG.VIOLET}{WEIGHT_RANGE}{color.Color.END}")
    print(f"\nTrain accuracy: - {adaline.train_accuracy}")
    print(f"Iterations: ----- {adaline.iterations}")
    print(f"Weights: -------- {weights[0]:.2f} | {weights[1]:.2f} | {weights[2]:.2f}")
    adaline.test(x_test, y_test)
    print(f"\nTest accuracy: -- {adaline.train_accuracy}")
    if SHOW_PLOT:
        plot_result(x_train, y_train, weights)


if __name__ == '__main__':
    main()





