import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Embedding
from keras.utils import pad_sequences
from keras.metrics import BinaryAccuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EPOCHS = 10
BATCH_SIZE = 128
ACTIVATION = 'tanh'

SAVE_PATH = 'experiments.json'


def load_data(padding_max_len, top_words, skip_top: int = 0):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        path="imdb.npz",
        num_words=top_words,
        skip_top=skip_top
    )
    x_train = pad_sequences(x_train, padding='post', maxlen=padding_max_len)
    x_test = pad_sequences(x_test, padding='post', maxlen=padding_max_len)
    return x_train, y_train, x_test, y_test


# 88584 - tyle jest różnych słów
# 2494 - maksymalna długość opini z paddingiem
# 1051 - maksymalna liczba słów w opini po splicie
def make_rnn(embedding_len: int, input_len: int, rnn_layer, mask=True):
    model = Sequential([
        Embedding(input_dim=88585, output_dim=embedding_len, input_length=input_len, mask_zero=mask),
        rnn_layer,
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mean_squared_error', BinaryAccuracy(), TruePositives(), TrueNegatives(), FalsePositives(),
                           FalseNegatives()])
    return model


def print_opinion(index_range, opinions):
    index = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
    reverse_index = {value: key for key, value in index.items()}
    mx = 0
    for opinion in opinions[index_range]:
        decoded = " ".join([reverse_index.get(i - 3, '') for i in opinion]).strip()
        print(decoded)
        l = len(decoded.split(' '))
        if l > mx:
            mx = l
        print("Total len: ", len(decoded.split(' ')))
    print(f"Max len: {mx}")


def embedding_space_test(space: int):
    layers = [SimpleRNN(units=64, activation=ACTIVATION), LSTM(units=64, activation=ACTIVATION)]
    padding_max_len = 800
    top_words = 500
    skip_top = 20
    x_train, y_train, x_test, y_test = load_data(padding_max_len, top_words=top_words, skip_top=skip_top)
    with open(SAVE_PATH, 'r', encoding='utf-8') as file:
        store = json.load(file)
    for layer in layers:
        model = make_rnn(space, padding_max_len, layer)
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose='auto',
                            workers=16, use_multiprocessing=True, validation_split=0.3)
        store[f'Embedding_{space}_{layer.__class__.__name__}'] = history.history
        with open(SAVE_PATH, 'w', encoding='utf-8') as file:
            json.dump(store, file)


def masking_test():
    masking = [True, False]
    layers = [SimpleRNN(units=64, activation=ACTIVATION), LSTM(units=64, activation=ACTIVATION)]
    padding_max_len = 800
    top_words = 500
    skip_top = 20
    x_train, y_train, x_test, y_test = load_data(padding_max_len, top_words=top_words, skip_top=skip_top)
    with open(SAVE_PATH, 'r', encoding='utf-8') as file:
        store = json.load(file)
    for mask in masking:
        for layer in layers:
            model = make_rnn(20, padding_max_len, layer, mask=mask)
            history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose='auto',
                                workers=16, use_multiprocessing=True, validation_split=0.3)
            store[f'Masking_{mask}_{layer.__class__.__name__}'] = history.history
            with open(SAVE_PATH, 'w', encoding='utf-8') as file:
                json.dump(store, file)


def padding_test():
    paddings = [100, 1000, 2494]
    layers = [SimpleRNN(units=64, activation=ACTIVATION), LSTM(units=64, activation=ACTIVATION)]
    top_words = 500
    skip_top = 20
    with open(SAVE_PATH, 'r', encoding='utf-8') as file:
        store = json.load(file)
    for padding in paddings:
        x_train, y_train, x_test, y_test = load_data(padding, top_words=top_words, skip_top=skip_top)
        for layer in layers:
            model = make_rnn(20, padding, layer, mask=True)
            history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose='auto',
                                workers=16, use_multiprocessing=True, validation_split=0.3)
            store[f'Padding_{padding}_{layer.__class__.__name__}'] = history.history
            with open(SAVE_PATH, 'w', encoding='utf-8') as file:
                json.dump(store, file)


if __name__ == '__main__':
    padding_test()
    # masking_test()
    # spaces = [2, 10, 50]
    # for space in spaces:
    #     embedding_space_test(space)
    # x_train, y_train, x_test, y_test = load_data(None, top_words=None, skip_top=0)
    # print(x_train.shape)
    # print_opinion(range(x_test.shape[1]), x_test)
    # d = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
    # for key, value in sorted(d.items(), key=lambda x: x[1]):
    #     if value < 300:
    #         print(f"{key}: {value}")
