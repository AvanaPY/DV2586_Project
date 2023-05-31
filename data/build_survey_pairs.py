# Modules 
import numpy as np
import matplotlib.pyplot as plt
import emnist

# Munctions
from typing import *

space = np.array([[0 for i in range(28)] for i in range(28)]) # Constant for "empty character", or "space"

def create_pairs(n:int=10, num_pairs:int=2):
    """
    Function for creating [original_image, generated_image] sets to use in survey.

    ### Parameters
    n: int
        The number of unique classes to return. Default is 10. 
    num_pairs: int
        The number of pairs per unique class to generate. Default is 2

    ### Returns
    xr: list[list[list[np.ndarray]]]
        A list of [original_image, generated_image] pairs, where each item in each 
        pair is a list of ndarrays representing images.
    yr: list[int]
        A list of all labels in the same order of the pairs in `xr` for mapping.
    """

    # Make sure we don't want more classes than we have in the dataset.
    if n > 26:
        return "There are only 26 classes in the dataset!"
    
    # Load emnist dataset (will download it if it does not exist locally).
    x, y = [np.array(lst) for lst in emnist.extract_training_samples('letters')]
    
    # Get n unique labels from the dataset.
    uniques = []
    xr, yr = [[] for _ in range(2)]
    for i in range(len(y)):
        if (y[i] not in uniques and len(uniques) <= n):
            uniques.append(y[i])
        # If we have reached the target amount, we can leave the loop.
        if len(uniques) >= n:
            break
    
    # Get num_pairs pairs for each unique label.
    dic = {} # Used to keep track on how many items we have of each .
    for i in range(len(y)):
        if y[i] in uniques:
            if y[i] not in dic:
                dic[y[i]] = 0
            
            # If we have enough of the current label, we can simply continue looking.
            if dic[y[i]] == num_pairs:
                continue
            dic[y[i]] += 1

            # Add the label to the return y list.
            yr.append(y[i])

            # Add the image on spot i in x and generate an image with the same target label.
            xr.append([x[i], _local_get_generated_character(y[i])])
        
        # If we have enough images (which is: the number of classes * the number of images per 
        # class we want), we break the loop
        if sum(dic.values()) == n * num_pairs:
            print(f'Found {num_pairs} of each class in\n{uniques}\nTotal number of ids: {len(yr)}\nTotal number of pairs: {len(xr)}')
            break

    return xr, yr

def _local_get_generated_character(id:int):
    return space

if __name__ == "__main__":
    pairs, classes = create_pairs(n=10, num_pairs=2)

