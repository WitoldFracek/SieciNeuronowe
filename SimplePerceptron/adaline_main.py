from adaline_perceptron import Adaline
from adaline_utils import output_mapping, signum_mapping
from perceptron_utils import AND_VALUES, OR_VALUES, XOR_VALUES, generate_training_set
import numpy as np

SET_SIZE = 1000
TRAIN_PERCENTAGE = 0.8
ITERATIONS = 120
ALPHA = 0.0001

if __name__ == '__main__':
    train_last_index = int(SET_SIZE * TRAIN_PERCENTAGE)
    x_ext, y_ext = generate_training_set(SET_SIZE, AND_VALUES)
    x_train, x_test = x_ext[..., :train_last_index], x_ext[..., train_last_index:]
    y_train, y_test = y_ext[..., :train_last_index], y_ext[..., train_last_index:]
    # adaline = Adaline(x_train, y_train, input_size=3, output_size=1, output_mapping=output_mapping(0))
    adaline = Adaline(x_train, y_train, input_size=3, output_size=1, output_mapping=np.vectorize(lambda x: 1 if x > 0 else 0))
    adaline.train(ITERATIONS, ALPHA)
    print(adaline.train_accuracy)
    print(adaline.weights / adaline.weights[0][0])

    adaline.test(x_test, y_test)
    print(adaline.test_accuracy)





