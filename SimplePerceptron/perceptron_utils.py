from enum import Enum

import numpy as np
import random


AND_VALUES = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
OR_VALUES = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
XOR_VALUES = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]


class ActFun(Enum):
    UNIPOLAR = 1
    BIPOLAR = 2


def generate_training_set(size, logic_values):
    ret_x = [np.array((1, x1, x2)) for x1, x2, _ in logic_values]
    ret_y = [y for _, _, y in logic_values]
    for _ in range(size - 4):
        x1, x2, y = random.choice(logic_values)
        sign1 = random.choice([-1, 1])
        sign2 = random.choice([-1, 1])
        new_x1 = x1 + sign1 * random.random() * 0.001
        new_x2 = x2 + sign2 * random.random() * 0.001
        ret_x.append(np.array((1, new_x1, new_x2)))
        ret_y.append(y)
    return np.column_stack(ret_x), np.asarray(ret_y).reshape(1, -1)


@np.vectorize
def unipolar_activation(value: float, theta=0) -> int:
    return 1 if value > theta else 0


@np.vectorize
def bipolar_activation(value: float, theta=0) -> int:
    return 1 if value > theta else -1


# def step_function_delta(predicted: np.ndarray, expected: np.ndarray, x: np.ndarray):
#     return (expected - predicted).dot(x.T)
