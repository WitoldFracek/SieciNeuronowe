from enum import Enum

import numpy as np
import random
import matplotlib.pyplot as pyl


AND_VALUES = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
OR_VALUES = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
XOR_VALUES = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]


class ActFun(Enum):
    UNIPOLAR = 1
    BIPOLAR = 2


def logic_pair(x1, x2, operator):
    if operator(x1, x2) == 0:
        return 1, 0
    else:
        return 0, 1


def operator_and(x1: float, x2: float) -> int:
    i1 = round(x1)
    i2 = round(x2)
    return min(i1, i2)


def operator_or(x1: float, x2: float) -> int:
    i1 = round(x1)
    i2 = round(x2)
    return max(i1, i2)


def operator_xor(x1: float, x2: float) -> int:
    i1 = round(x1)
    i2 = round(x2)
    return (i1 + i2) % 2


def generate_set(size, logic_values, operator):
    ret_x = [np.array((x1, x2)) for x1, x2, _ in logic_values]
    ret_y = [np.array(logic_pair(x1, x2, operator)) for x1, x2, y in logic_values]
    for _ in range(size - 4):
        x1, x2, y = random.choice(logic_values)
        sign1 = random.choice([-1, 1])
        sign2 = random.choice([-1, 1])
        new_x1 = x1 + sign1 * random.random() * 0.001
        new_x2 = x2 + sign2 * random.random() * 0.001
        ret_x.append(np.array((new_x1, new_x2)))
        ret_y.append(np.array(logic_pair(new_x1, new_x2, operator)))
    return np.column_stack(ret_x), np.column_stack(ret_y)


def generate_set_flat(size, logic_values):
    ret_x = [np.array((x1, x2)) for x1, x2, _ in logic_values]
    ret_y = [y for _, _, y in logic_values]
    for _ in range(size - 4):
        x1, x2, y = random.choice(logic_values)
        sign1 = random.choice([-1, 1])
        sign2 = random.choice([-1, 1])
        new_x1 = x1 + sign1 * random.random() * 0.001
        new_x2 = x2 + sign2 * random.random() * 0.001
        ret_x.append(np.array((new_x1, new_x2)))
        ret_y.append(y)
    return np.column_stack(ret_x), np.array(ret_y).reshape((1, -1))


@np.vectorize
def unipolar_activation(value: float, theta=0) -> int:
    return 1 if value > theta else 0


@np.vectorize
def bipolar_activation(value: float, theta=0) -> int:
    return 1 if value > theta else -1


def plot_result(x: np.ndarray, y: np.ndarray, w: np.ndarray, plot_range=(0, 1), title=""):
    for i in range(x.shape[1]):
        if y[0][i] == 1:
            pyl.scatter([x[2][i]], [x[1][i]], c='b')
        else:
            pyl.scatter([x[2][i]], [x[1][i]], c='r')
    plot_min, plot_max = plot_range
    x_all = np.linspace(plot_min, plot_max, 10)
    pyl.plot(x_all, -w[1] / w[2] * x_all - w[0] / w[2])
    pyl.title(title)
    pyl.grid(linestyle='--', linewidth=0.5)
    pyl.axhline(y=0, color='k')
    pyl.axvline(x=0, color='k')
    pyl.show()

