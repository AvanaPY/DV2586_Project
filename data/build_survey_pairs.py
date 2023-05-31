# Munctions
from typing import *
# Modules 
import numpy as np
import matplotlib.pyplot as plt
import emnist
import tensorflow as tf
import keras 
from utils.noise import chars_to_noise
space = np.array([[0 for i in range(28)] for i in range(28)]) # Constant for "empty character", or "space"

def create_pairs(generator : keras.Model,
                 char_map : Dict[int, str],
                 n_class : int = 10, 
                 n_pairs : int = 2):
    """
    Function for creating [original_image, generated_image] sets to use in survey.

    ### Parameters
    n_class: int
        The number of unique classes to return. Default is 10. 
    n_pairs: int
        The number of pairs per unique class to generate. Default is 2

    ### Returns
    xr: list[list[list[np.ndarray]]]
        A list of [original_image, generated_image] pairs, where each item in each 
        pair is a list of ndarrays representing images.
    yr: list[int]
        A list of all labels in the same order of the pairs in `xr` for mapping.
    """


    # Make sure we don't want more classes than we have in the dataset.
    if n_class > 26:
        return "There are only 26 classes in the dataset!"
    
    # Load emnist dataset (will download it if it does not exist locally).
    x, y = [np.array(lst) for lst in emnist.extract_training_samples('letters')]
    
    # Get n unique labels from the dataset.
    uniques = []
    xr, yr = [], []
    for i in range(len(y)):
        if (y[i] not in uniques and len(uniques) <= n_class):
            uniques.append(y[i])
        # If we have reached the target amount, we can leave the loop.
        if len(uniques) >= n_class:
            break
    
    # Get num_pairs pairs for each unique label.
    dic = {} # Used to keep track on how many items we have of each .
    for i in range(len(y)):
        if y[i] in uniques:
            if y[i] not in dic:
                dic[y[i]] = 0
            
            # If we have enough of the current label, we can simply continue looking.
            if dic[y[i]] == n_pairs:
                continue
            dic[y[i]] += 1

            # Add the label to the return y list.
            yr.append(y[i])

            # Add the image on spot i in x and generate an image with the same target label.
            noise = _local_get_generated_character(generator, char_map, y[i])
            xr.append([x[i], noise])
        
        # If we have enough images (which is: the number of classes * the number of images per 
        # class we want), we break the loop
        if sum(dic.values()) == n_class * n_pairs:
            print(f'Found {n_pairs} of each class in\n{uniques}\nTotal number of ids: {len(yr)}\nTotal number of pairs: {len(xr)}')
            break

    v = [(x, y) for x, y in zip(xr, yr)]
    v = sorted(v, key=lambda x : x[1])
    return [x for x, _ in v], [y for _, y in v]

def _local_get_generated_character(generator : keras.Model,
                                   char_map : Dict[int, str], 
                                   id : int,
                                   n_classes : int = 27,
                                   noise_dim : int = 1):
    c = char_map[id]
    noise = chars_to_noise(c, char_map, noise_dim, n_classes)
    image = generator(noise)
    image = tf.squeeze(image)
    return image

if __name__ == "__main__":
    from data import build_categorical_dataset
    _, char_map = build_categorical_dataset()
    pairs, classes = create_pairs(char_map, n_class=10, n_pairs=2)

    plt.figure(figsize=(5, 6))
    for i, ((x, y), c) in enumerate(zip(pairs, classes)):
        plt.subplot(5, 6, i+1)
        plt.imshow(x, cmap='gray')
        plt.title(char_map[c])
        
    plt.show()