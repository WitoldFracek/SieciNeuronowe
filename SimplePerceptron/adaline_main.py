from adaline_perceptron import Adaline
from adaline_utils import generate_training_set, output_mapping
from perceptron_utils import AND_VALUES, OR_VALUES, XOR_VALUES

SET_SIZE = 10
TRAIN_PERCENTAGE = 0.8
ITERATIONS = 10_000
ALPHA = 0.2

if __name__ == '__main__':
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, AND_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    adaline = Adaline(x_train, y_train, input_size=3, output_size=1, output_mapping=output_mapping())
    adaline.train(ITERATIONS, ALPHA)
    print(adaline.train_accuracy)





