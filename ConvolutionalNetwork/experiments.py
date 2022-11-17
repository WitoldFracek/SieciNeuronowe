import tensorflow as tf
import matplotlib.pyplot as plt
from keras.metrics import Accuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, MeanSquaredError, \
    CategoricalAccuracy
import numpy as np
from main import prepare_sets
import os
import json

# Network params
CLASSES_COUNT = 10
INPUT_SHAPE = (28, 28, 1)
FILTERS = 32
KERNEL_SIZE = (3, 3)
POOLING_SIZE = (2, 2)
STEP = 2
HIDDEN_LAYER_NEURONS = 128
ACTIVATION = 'relu'

# Training params
EPOCHS = 3
BATCH_SIZE = 30
STEPS_PER_EPOCH = None
METRICS = [CategoricalAccuracy(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(),
           MeanSquaredError()]
METRICS_NAMES = ['loss'] + [x.__class__.__name__ for x in METRICS]


def pooling_type_test(x_train, y_train):
    pool_sizes = [2, 3, 4]
    for pool_size in pool_sizes:
        pooling_type = [
            tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=1),
            tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1),
        ]
        for pooling_layer in pooling_type:
            print(f'Preparing model with {pooling_layer.__class__.__name__}: pool_size = {pool_size}')
            model = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=INPUT_SHAPE),
                    tf.keras.layers.Conv2D(FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION),
                    pooling_layer,
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=ACTIVATION),
                    tf.keras.layers.Dense(CLASSES_COUNT, activation="softmax"),
                ]
            )
            model.compile(loss="categorical_crossentropy", optimizer="adam",
                          metrics=METRICS)
            print(f'Fitting model with {pooling_layer.__class__.__name__}: pool_size = {pool_size}')
            history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                      workers=4, use_multiprocessing=True, verbose=1, validation_split=0.33)
            print(history.history)
            model_name = f'Pool_{pooling_layer.__class__.__name__}_size_{pool_size}'
            store = dict()
            with open('experiments.json', 'r', encoding='utf-8') as file:
                store = json.load(file)
                store[model_name] = history.history
            with open('experiments.json', 'w', encoding='utf-8') as file:
                json.dump(store, file)


def prepare_models():
    x_train, y_train, x_test, y_test = prepare_sets()
    pooling_type_test(x_train, y_train)


def plot_history_for_model(model):
    acc = model.history['accuracy']
    plt.plot(acc)
    plt.show()


if __name__ == '__main__':
    prepare_models()
    # model = tf.keras.models.load_model(os.path.join('experiments', 'Pool_AveragePooling2D_size_2'))
    # plot_history_for_model(model)

