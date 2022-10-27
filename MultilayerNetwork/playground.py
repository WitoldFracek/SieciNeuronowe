import numpy as np


#keras.datasets
# from keras.datasets import mnist

def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))

arr = np.array([
    [1, 2, 3],
    [0, 0, 3],
    [5, 1, 3]
])

# sm = np.column_stack([softmax(column) for column in arr[...].T])
# print(sm)

xs = [x for x in range(10)]
for elem in xs[-2::-1]:
    print(elem)


