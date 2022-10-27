import numpy as np
from utils import softmax, relu

#keras.datasets
# from keras.datasets import mnist

arr = np.array([
    [-1, -2, -3, -4],
    [0, 0, 3, 0],
    [5, 1, 3, 1]
])

print(softmax(arr))


