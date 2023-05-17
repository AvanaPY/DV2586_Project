# Modules
import numpy as np
import matplotlib.pyplot as plt
import emnist

# Meh functions
from typing import *

# Cool functions
#from data.data import *

def _local_get_img():
    x, y = emnist.extract_training_samples('balanced')
    x = x[0:50]
    y = y[0:50]
    xt = []
    yt = []
    t = -1
    for i, item in enumerate(x):
        if i % 10 == 0:
            xt.append([item])
            t += 1
        else:
            xt[t].append(item)
    t = -1
    for i, item in enumerate(y):
        if i % 10 == 0:
            yt.append([item])
            t += 1
        else:
            yt[t].append(item)
    return xt, yt

def horizontal(lst:list[np.ndarray]):
    full_row = 0
    for i, item in enumerate(lst):
        if i == 0:
            full_row = item
        else:
            full_row = np.concatenate((full_row, item), axis=1)
    return full_row

def vertical(lst:list[list[np.ndarray]]):
    space = np.array([[0 for i in range(28)] for i in range(28)])
    len_lst = [len(x) for x in lst]
    finale = 0
    for i, item in enumerate(lst):
        if len(item) < max(len_lst):
            for j in range(max(len_lst) - len(item)):
                item.append(space)
        row = horizontal(item)
        if i == 0:
            finale = row
            print(f'{i}, {len(finale)}')
        else:
            finale = np.concatenate((finale, row), axis=0)
            print(f'{i}, {len(finale)}')
    print(f'Final, {len(finale)}')
    return finale

# Test section
if __name__ == "__main__":
    x, y = _local_get_img()
    del x[4][1]
    concat = vertical(x)
    print(concat.shape)
    plt.imshow(concat, interpolation="nearest", cmap="gray")
    plt.show()