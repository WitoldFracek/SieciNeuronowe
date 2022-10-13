from adaline_perceptron import Adaline
from perceptron_utils import AND_VALUES, OR_VALUES, XOR_VALUES, generate_training_set, plot_result

SET_SIZE = 1000
TRAIN_PERCENTAGE = 0.8
MIN_ERR = 0.001
ALPHA = 0.01

if __name__ == '__main__':
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, AND_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    adaline = Adaline(x_train, y_train, input_size=3, output_size=1, output_mapping=lambda x: 1 if x > 0 else 0)
    adaline.train(MIN_ERR, ALPHA)
    weights = adaline.weights[0] / adaline.weights[0][0]
    print(f"\nModel: {adaline.__class__.__name__.upper()}")
    print(f"Initial set size: {SET_SIZE} | Training size {train_last_index}")
    print(f"Train accuracy: - {adaline.train_accuracy}")
    print(f"Iterations: ----- {adaline.iterations}")
    print(f"Weights: -------- {weights[0]:.2f} | {weights[1]:.2f} | {weights[2]:.2f}")
    adaline.test(x_test, y_test)
    print(f"Test accuracy: -- {adaline.train_accuracy}")
    plot_result(x_train, y_train, weights)





