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
EPOCHS = 20
BATCH_SIZE = 100
STEPS_PER_EPOCH = None
METRICS = [CategoricalAccuracy(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(), MeanSquaredError()]
METRICS_NAMES = ['loss'] + [x.__class__.__name__ for x in METRICS]


def json_save(name, history):
    with open('experiments.json', 'r', encoding='utf-8') as file:
        store = json.load(file)
        store[name] = history
    with open('experiments.json', 'w', encoding='utf-8') as file:
        json.dump(store, file)


def cnn_vs_mlp(x_train, y_train):
    mlp1 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation=ACTIVATION),
        tf.keras.layers.Dense(10)
    ])
    mlp1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)

    mlp2 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(60, activation=ACTIVATION),
        tf.keras.layers.Dense(50, activation=ACTIVATION),
        tf.keras.layers.Dense(10)
    ])
    mlp2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)

    cnn1 = tf.keras.Sequential(
        [
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=(3, 3), activation=ACTIVATION),
            tf.keras.layers.MaxPooling2D(pool_size=POOLING_SIZE, strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=ACTIVATION),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    cnn1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)

    cnn2 = tf.keras.Sequential(
        [
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=(3, 3), activation=ACTIVATION),
            tf.keras.layers.AveragePooling2D(pool_size=POOLING_SIZE, strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=ACTIVATION),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    cnn2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)

    cnn3 = tf.keras.Sequential(
        [
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=(5, 5), activation=ACTIVATION),
            tf.keras.layers.MaxPooling2D(pool_size=POOLING_SIZE, strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=ACTIVATION),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    cnn3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)

    # fit models
    history = mlp1.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                        workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'mlp_vs_cnn_mlp1', history.history)

    history = mlp2.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                       workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'mlp_vs_cnn_mlp2', history.history)

    history = cnn1.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                       workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'mlp_vs_cnn_cnn1', history.history)

    history = cnn2.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                       workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'mlp_vs_cnn_cnn2', history.history)

    history = cnn3.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                       workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'mlp_vs_cnn_cnn3', history.history)


def pooling_type_test(x_train, y_train):
    pool_sizes = [2, 3, 4, 5]
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
            model_name = f'Pool_{pooling_layer.__class__.__name__}_size_{pool_size}'
            json_save(model_name, history.history)


def augmentation_test(x_train, y_train):
    random_flip = tf.keras.Sequential([
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=(3, 3), activation=ACTIVATION),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=ACTIVATION),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    random_flip.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)
    random_translation = tf.keras.Sequential([
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=(3, 3), activation=ACTIVATION),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=ACTIVATION),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    random_translation.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)
    random_rotation = tf.keras.Sequential([
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.RandomRotation(factor=0.2),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=(3, 3), activation=ACTIVATION),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=ACTIVATION),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    random_rotation.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)
    random_flip_and_translation = tf.keras.Sequential([
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=(3, 3), activation=ACTIVATION),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=ACTIVATION),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    random_flip_and_translation.compile(loss="categorical_crossentropy", optimizer="adam", metrics=METRICS)

    # model fit
    history = random_flip.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                              workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'augmentation_flip', history.history)

    history = random_rotation.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                              workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'augmentation_rotation', history.history)

    history = random_translation.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                       workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'augmentation_translation', history.history)

    history = random_flip_and_translation.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
                       workers=16, use_multiprocessing=True, verbose=1, validation_split=0.33)
    json_save(f'augmentation_flip_translation', history.history)


def prepare_models():
    x_train, y_train, x_test, y_test = prepare_sets()
    cnn_vs_mlp(x_train, y_train)
    pooling_type_test(x_train, y_train)
    augmentation_test(x_train, y_train)


def plot_history_for_model(model):
    acc = model.history['accuracy']
    plt.plot(acc)
    plt.show()


if __name__ == '__main__':
    prepare_models()

