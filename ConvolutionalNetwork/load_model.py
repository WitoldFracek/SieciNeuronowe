import tensorflow as tf
from keras.datasets.mnist import load_data
import numpy as np
from main import SAVE_PATH, prepare_sets, METRICS_NAMES

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_sets()
    model = tf.keras.models.load_model(SAVE_PATH)
    for name, score in zip(METRICS_NAMES, model.evaluate(x_test, y_test, verbose=0)):
        print(f"{name}: {score}")



