import numpy as np
import random
from keras.datasets import mnist
import matplotlib.pyplot as pyl


def show_random(x: np.ndarray, predictions: np.ndarray, labels: np.ndarray, nbest=1):
    pred_col, pred_lab, x_rnd = get_random_prediction(predictions, labels, x)
    label = transform_row_to_number(pred_lab)[0][0]
    column_best_fits = get_best_fits_in_column(pred_col, n=nbest)
    pyl.title(f"Predicted: {' | '.join([f'{i}: {v * 100:.2f}%' for i, v in column_best_fits])}\nActual: {label}")
    img = x_rnd.reshape((28, 28)).T
    pyl.imshow(img, cmap='gray')
    pyl.show()


def show_bad(pred_col, label, x_rnd):
    column_best_fits = get_best_fits_in_column(pred_col)
    pyl.title(f"Predicted: {' | '.join([f'{i}: {v * 100:.2f}%' for i, v in column_best_fits])}\nActual: {label}")
    img = x_rnd.reshape((28, 28)).T
    pyl.imshow(img, cmap='gray')
    pyl.show()


def false_values_generator(x: np.ndarray, predictions: np.ndarray, labels: np.ndarray):
    counter = 0
    while counter < predictions.shape[1]:
        pred_col, pred_lab, x_rnd = get_prediction(counter, predictions, labels, x)
        label = transform_row_to_number(pred_lab)[0][0]
        cipher = np.argmax(pred_col)
        if label != cipher:
            yield pred_col, label, x_rnd
        counter += 1


def get_random_prediction(predictions: np.ndarray, labels: np.ndarray, x: np.ndarray):
    max_ind = predictions.shape[1]
    index = random.randint(0, max_ind - 1)
    return predictions.T[index], labels.T[index], x.T[index]


def get_prediction(index: int, predictions: np.ndarray, labels: np.ndarray, x: np.ndarray):
    return predictions.T[index], labels.T[index], x.T[index]


def shuffle_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zipped = list(zip(x.T, y.T))
    random.shuffle(zipped)
    x_shuffled, y_shuffled = unzip_arrays(zipped)
    return x_shuffled, y_shuffled


def make_batch(batch_size: int, x: np.ndarray, y: np.ndarray):
    zipped = list(zip(x.T, y.T))
    sample = random.sample(zipped, batch_size)
    x_sample, y_sample = unzip_arrays(sample)
    return x_sample, y_sample


def unzip_arrays(zipped: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    x_unzipped = np.column_stack([x for x, _ in zipped])
    y_unzipped = np.column_stack([y for _, y in zipped])
    return x_unzipped, y_unzipped


def prediction_index_from_column(column: np.ndarray, n: int = 1):
    ind = np.argpartition(column, -n)[-n:]
    return ind


def get_n_best_predictions(classes: np.ndarray, n: int = 1):
    xs = [prediction_index_from_column(col, n) for col in classes.T]
    return np.column_stack(xs)


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.T.reshape(28 * 28, -1) / 255
    x_test = x_test.T.reshape(28 * 28, -1) / 255

    temp = []
    for elem in y_train:
        col = transform_row_to_one_hit(elem)
        temp.append(col)
    new_y_train = np.column_stack(temp)

    temp = []
    for elem in y_test:
        col = transform_row_to_one_hit(elem)
        temp.append(col)
    new_y_test = np.column_stack(temp)

    return x_train, new_y_train, x_test, new_y_test


def transform_row_to_one_hit(index):
    ret = np.zeros((10, 1))
    ret[index] = 1
    return ret


def transform_row_to_number(row):
    index = np.where(row == 1)
    return index


def get_best_fits_in_column(column: np.ndarray, n: int = 1):
    ind = prediction_index_from_column(column, n=n)
    ind_list = ind.tolist()
    values_list = column[ind].tolist()
    index_value_pairs = zip(ind_list, values_list)
    index_value_pairs = sorted(index_value_pairs, key=lambda x: x[1], reverse=True)
    return index_value_pairs


def best_predictions_tuple_list(predictions: np.ndarray, labels: np.ndarray, n: int = 1) -> list[tuple]:
    ret = []
    best_ind = get_n_best_predictions(predictions, 3)
    for ind, col, label in zip(best_ind.T, predictions.T, labels.T):
        ind_list = ind.tolist()
        values_list = col[ind].tolist()
        index_value_pairs = zip(ind_list, values_list)
        index_value_pairs = sorted(index_value_pairs, key=lambda x: x[1], reverse=True)
        ret.append((index_value_pairs, label[0]))
    return ret


