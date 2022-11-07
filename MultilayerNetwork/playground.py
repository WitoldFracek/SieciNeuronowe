import numpy as np
from functions import softmax

a = np.array([[1, 2, 2, 4, 4],
              [1, 1, 1, 1, 1]]).T
l = np.array([[0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0]]).T

print(a)
print(softmax(a))

