import numpy as np
from data_transform import get_n_best_predictions, get_best_fits_in_column, show_random

a = np.array([[0.1, 0.11, 0.09, 1, 0.5],
              [1, 0.01, 0.2, 0.1, 0.3]]).T
l = np.array([[0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0]]).T

show_random(a, l)


