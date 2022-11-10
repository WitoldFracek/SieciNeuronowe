import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.metrics import Accuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, MeanSquaredError, CategoricalAccuracy
import numpy as np

# saves
SAVE_PATH = 'model'

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
EPOCHS = 10
BATCH_SIZE = 100
STEPS_PER_EPOCH = None
METRICS = [CategoricalAccuracy(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives(), MeanSquaredError()]
METRICS_NAMES = ['loss'] + [x.__class__.__name__ for x in METRICS]


def prepare_sets():
    (x_train, y_train), (x_test, y_test) = load_data()

    # Normalise images
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert classes to binary matrix
    y_train = tf.keras.utils.to_categorical(y_train, CLASSES_COUNT)
    y_test = tf.keras.utils.to_categorical(y_test, CLASSES_COUNT)

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = prepare_sets()

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION),
            tf.keras.layers.MaxPooling2D(pool_size=POOLING_SIZE, strides=STEP),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation=ACTIVATION),
            tf.keras.layers.Dense(CLASSES_COUNT, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=METRICS)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
              workers=4, use_multiprocessing=True, verbose=1)
    model.save(SAVE_PATH)
    score = model.evaluate(x_test, y_test, verbose=0)
    for name, s in zip(METRICS_NAMES, score):
        print(f"{name}: {s}")


if __name__ == '__main__':
    main()

