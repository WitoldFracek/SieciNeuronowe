import numpy as np

a = np.array([[0.1, 0.11, 0.09, 1, 0.5],
              [2, 0.01, 0.2, 0.1, 0.3]]).T
l = np.array([[0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0]]).T

for _ in range(10):
    print(np.random.randn(1, 6))
    print(np.random.normal(1, 1, size=(1, 6)))
    print()

