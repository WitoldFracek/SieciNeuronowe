import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Embedding, ZeroPadding1D, Masking
from sklearn.metrics import mean_squared_error
from keras.utils import pad_sequences
import matplotlib.pyplot as plt

NUM_WORDS = None  # default: None
SKIP_TOP = 0  # default: 0
MAX_LEN = None  # default: None
START_CHAR = 1  # default: 1
OOV_CHAR = 2

BATCH_SIZE = 128

EMBEDDING_SPACE = 100
ACTIVATION = 'tanh'


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        path="imdb.npz",
        num_words=NUM_WORDS,
        skip_top=SKIP_TOP
    )
    print(y_train.shape)
    return pad_sequences(x_train, padding='post'), y_train, pad_sequences(x_test, padding='post'), y_test
    # return x_train, y_train, x_test, y_test


# 88584
def make_rnn():
    model = Sequential([
        Embedding(input_dim=90_000, output_dim=64, mask_zero=True),
        SimpleRNN(units=128, activation='tanh'),
        # LSTM(128),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model


def main():
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape)
    model = make_rnn()
    history = model.fit(x_train, y_train, epochs=1, verbose="auto", workers=4,
                        use_multiprocessing=True, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
    # d = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
    # print(d[max(d, key=lambda x: d[x])])
    # print(d['the'])
    # print(d['movie'])
    # for i, (key, value) in enumerate(d.items()):
    #     if value == 2:
    #         print(key)

