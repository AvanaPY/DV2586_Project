from typing import *
import numpy as np

empty_character = np.zeros(shape=(28, 28))

def create_categorised_noise(noise_dims : int, n_classes : int):
    """
        Generates one noise image per class. 
        
        ## Parameters
        noise_dims : int
            The noise image dimensions.
        n_classes  : int
            The number of unique classes.
        
        ## Returns
        (noise, noise_y) : Tuple[np.ndarray, np.ndarry]
            Tuple of noise images and noise labels.
    """
    n_classes -= 1
    noise = np.zeros(shape=(n_classes, n_classes, noise_dims))
    noise_y = np.arange(n_classes) + 1
    for i in range(n_classes):
        noise[i,i,:] = np.random.uniform(low=-1, high=1,size=noise_dims)
    return noise, noise_y

def str_to_noise(s : str, char_map : Dict[str, int], noise_dims : int, n_classes : int):
    words = s.split(' ')
    noises = []
    for word in words:
        noise = chars_to_noise(word, char_map, noise_dims, n_classes)
        noises.append(noise)
    return noises

def chars_to_noise(chars : str, char_map : Dict[str, int], noise_dims : int, n_classes : int):
    n_classes -= 1
    n_chars = len(chars)
    noise = np.zeros(shape=(n_chars, n_classes, noise_dims))
    for i, char in enumerate(chars):
        if char != ' ':
            ci = char_map[char] - 1
            noise[i,ci,:] = np.random.uniform(low=-1, high=1, size=noise_dims)
    
    return noise

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    x, y = create_categorised_noise(7, 27)
    print(x.shape)
    print(x)
    print(y)