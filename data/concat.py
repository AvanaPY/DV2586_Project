# Modules
import numpy as np
import matplotlib.pyplot as plt
import emnist

# Meh functions
from typing import *

def _local_get_img():
    x, y = [lst[0:50] for lst in emnist.extract_training_samples('balanced')]
    xt, yt = [[] for _ in range(2)]
    t = -1
    for i in range(len(x)):
        if i % 10 == 0:
            xt.append([x[i]])
            yt.append([y[i]])
            t += 1
        else:
            xt[t].append(x[i])
            yt[t].append(y[i])
    return xt, yt

def horizontal(lst:list[np.ndarray]):
    full_row = 0
    for i, item in enumerate(lst):
        full_row = item if i == 0 else np.concatenate((full_row, item), axis=1)
    return full_row

def vertical(lst:list[list[np.ndarray]]):
    space = np.array([[0 for i in range(28)] for i in range(28)])
    lrg = max([len(x) for x in lst])
    finale = 0
    for i, item in enumerate(lst):
        if len(item) < lrg:
            for j in range(lrg - len(item)):
                item.append(space)
        finale = horizontal(item) if i == 0 else np.concatenate((finale, horizontal(item)), axis=0)
    return finale

# Test section
if __name__ == "__main__":
    x, y = _local_get_img()
    del x[4][1]
    concat = vertical(x)
    plt.imshow(concat, interpolation="nearest", cmap="gray")
    plt.show()