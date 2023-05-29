##################################################
# Output utils module
#
# Note: Module works in "characters", defined as 
#       28x28 pixel images. All height and width 
#       definitions should be defined in number of 
#       characters, not in pixels
#
##################################################

# Modules
import numpy as np
import matplotlib.pyplot as plt
import emnist

# Munctions
from typing import *

space = np.array([[0 for i in range(28)] for i in range(28)]) # Constant for "empty character", or "space"


def _local_get_img(n:int=50):
    """
    Local function to load the `n` first samples from the emnist dataset.

    ### Parameters
    n: int 
        Number of samples to include. Default is `50`
    ### Returns
    xt: list[list[np.ndarray]] 
        The input values for the `n` first samples from the dataset.
    yt: list[list[np.ndarray]] 
        The labels for the `n` first samples from the dataset.
    """
    # Load emnist dataset (will download it if it does not exist locally)
    x, y = [lst[0:n] for lst in emnist.extract_training_samples('balanced')]
    xt, yt = [[] for _ in range(2)]
    t = -1
    # Seperate them into 10 character-long lists. Done to test functions such 
    # as horizontal and vertical, as well as the align functions
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
    """
    Function for combining a list of 28x28 sized iamges into a 28x(n*28) image 
    where n is the length of the list.

    ### Parameters
    lst: list[np.ndarray]
        A list of 28x28 `ndarrays` representing images.
    ### Returns
    full_row: np.ndarray
        An `ndarray` representing the concatenated image.
    """
    # concatenate every image in the list horizontally
    full_row = 0
    for i, item in enumerate(lst):
        full_row = item if i == 0 else np.concatenate((full_row, item), axis=1)
    return full_row

def vertical(lst:list[list[np.ndarray]], center:bool=False, right:bool=True):
    """
    Function for combining a matrix of 28x28 sized images into one image.

    ### Parameters
    lst: list[list[np.ndarray]]
        A list of lists containing 28x28 `ndarrays` representing images.
    center: bool
        A boolean representing if the text should be centered if the width of the image 
        is shorter than the page width.
    right: bool, define only if `center` is `False`
        A boolean that tells if we should align to left or right. Default is `True`.
    ### Returns
    finale: np.ndarray 
        An `ndarray` representing the concatenated image.
    """
    # Find the longest row in the list, this will be our "page width"
    page_width = max([len(x) for x in lst])
    
    # Append all iamgest vertically
    finale = 0
    for i, item in enumerate(lst):
        # If the list is shorter than the page width, insert blank spaces to 
        # either center or align the text left/right, depending on input
        if len(item) < page_width:
            item = centering(lst=item, diff=page_width - len(item)) if center else fill(lst=item, diff=page_width - len(item),right=right)
        # Turn the current list into one image, then concatenate vertically.
        finale = horizontal(item) if i == 0 else np.concatenate((finale, horizontal(item)), axis=0)
    return finale

def padding(img:np.ndarray, height:int, width:int):
    """
    Fuinction for padding the edges of an image.

    ### Parameters
    img: np.ndarray 
        An array representing the image to be appended.
    height int
        The number of characters the image should be in height (including original height).
    width: int
        The number of characters the image should be in width (including original width).
    ### Returns
    img: np.ndarray
        An `ndarray` representing the padded images.
    """
    # Check if image is horter or longer than the target height or width, 
    # in which case we simply do nothing
    if img.shape[0]/28 > height:
        print("Image's height is already larger than desired height.")
        return img 
    if img.shape[1]/28 > width:
        print("Image's width is already larger than desired width.")
        return img
    
    # Alternatively add rows of space characters top-bottom until the 
    # image reaches the target height
    for _ in range(height - int(img.shape[0]/28)):
        for i in range(int(img.shape[1]/28)):
            h = space if i == 0 else np.concatenate((h, space), axis=1)
        img = np.concatenate((img, h), axis=0) if img.shape[0]/28 % 2 == 0 else np.concatenate((h, img), axis=0)

    # Alternatively add rows of space characters right-left until the 
    # image reaches the target width
    for _ in range(width - int(img.shape[1]/28)):
        for i in range(int(img.shape[0]/28)):
            w = space if i == 0 else np.concatenate((w, space), axis=0)
        img = np.concatenate((img, w), axis=1) if img.shape[1]/28 % 2 == 0 else np.concatenate((w, img), axis=1)

    return img

def centering(lst:list[np.ndarray], diff:int):
    """
    Function for filling an uneven row to be centered.

    ### Parameters
    lst: list[np.ndarray]
        List of `ndarrays` to be filled.
    diff: int
        How many `space`s that are needed.
    ### Returns
    lst: list[np.ndarray]
        The centered list.
    """
    for i in range(diff):
        lst.append(space) if i % 2 == 0 else lst.insert(0, space)
    return lst
        
def fill(lst:list[np.ndarray], diff:int, right:bool=True):
    """
    Function for filling an uneven row to be aligned right or left.

    ### Parameters
    lst: list[np.ndarray]
        List of `ndarrays` to be filled.
    diff: int
        How many `space`s that are needed.
    right: bool
        A boolean that tells the function if we should align to left or right. Default is `True`.
    ### Returns
    lst: list
        The filled list.
    """
    for j in range(diff):
        lst.append(space) if right else lst.insert(0,space)
    return lst

# Test section
if __name__ == "__main__":
    x, y = _local_get_img()
    del x[3][2]
    del x[3][1]
    del x[4][1]
    concat = vertical(x, center=False, right=False)
    print(concat.shape)
    # concat = padding(concat, 8, 12)
    # print(concat.shape)
    plt.imshow(concat, interpolation="nearest", cmap="gray")
    plt.show()