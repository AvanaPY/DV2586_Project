##################################################
# Output utils module
#
# Note: Module works in "characters", 28x28 pixel 
#       images. All height and width definitions 
#       should be defined in number of characters,
#       not in pixels
#
##################################################

# Modules
import numpy as np
import matplotlib.pyplot as plt
import emnist

# Munctions
from typing import *

space = np.array([[0 for i in range(28)] for i in range(28)]) # Constant for "empty character", or "space"

"""
Local function to load the n first samples from the emnist dataset.

Input:
    n       (int): number of samples [default 50]
Output:
    xt     (list): the input values for the n first samples from the dataset.
    yt     (list): the labels for the n first samples fro mthe dataset.

"""
def _local_get_img(n:int=50):
    x, y = [lst[0:n] for lst in emnist.extract_training_samples('balanced')]
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

"""
Function for combining a list of 28x28 sized iamges into a 28x(n*28) immge 
where n is the length of the list.

Input:
    lst             (lsit): a list of 28x28 ndarrays representing images,
Output:
    full_row  (np.ndarray): an ndarray representing the concatenated image.

"""
def horizontal(lst:list[np.ndarray]):
    # concatenate every image in the list horizontally
    full_row = 0
    for i, item in enumerate(lst):
        full_row = item if i == 0 else np.concatenate((full_row, item), axis=1)
    return full_row

"""
Function for combining a list of 28x28 sized iamges into a 28x(n*28) immge 
where n is the length of the list.

Input:
    lst             (lsit): a list of 28x28 ndarrays representing images,
Output:
    finale    (np.ndarray): an ndarray representing the concatenated image.

"""
def vertical(lst:list[list[np.ndarray]]):
    # Find the longest row in the list, this will be our "page width"
    lrg = max([len(x) for x in lst])
    
    # Append all iamgest vertically
    finale = 0
    for i, item in enumerate(lst):
        # If the list is shorter than the page width, insert blank spaces next to it until it is
        if len(item) < lrg:
            for j in range(lrg - len(item)):
                item.append(space)
        
        # Turn the current list into one image, then concatenate vertically.
        finale = horizontal(item) if i == 0 else np.concatenate((finale, horizontal(item)), axis=0)
    return finale

"""
Fuinction for padding the edges of an image.

Input:
    img     (np.ndarray): an array representing the image to be appended.
    height         (int): the number of characters the image should be in height
    width   	   (int): the number of characters the image should be in width
Output:
"""
def padding(img:np.ndarray, height:int, width:int):
    if img.shape[0]/28 > height:
        print("Image's height is already larger than desired height.")
        return img 
    if img.shape[1]/28 > width:
        print("Image's width is already larger than desired width.")
        return img
    for _ in range(height - int(img.shape[0]/28)):
        for i in range(int(img.shape[1]/28)):
            h = space if i == 0 else np.concatenate((h, space), axis=1)
        img = np.concatenate((img, h), axis=0) if img.shape[0]/28 % 2 == 0 else np.concatenate((h, img), axis=0)

    for _ in range(width - int(img.shape[1]/28)):
        for i in range(int(img.shape[0]/28)):
            w = space if i == 0 else np.concatenate((w, space), axis=0)
        img = np.concatenate((img, w), axis=1) if img.shape[1]/28 % 2 == 0 else np.concatenate((w, img), axis=1)

    return img

# Test section
if __name__ == "__main__":
    x, y = _local_get_img()
    concat = vertical(x)
    print(concat.shape)
    padded = padding(concat, 8, 12)
    print(padded.shape)
    plt.imshow(padded, interpolation="nearest", cmap="gray")
    plt.show()