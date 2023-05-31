from typing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import keras
from model.generator import build_generator_model, build_discriminator_model
from utils.noise import chars_to_noise
from utils import vertical
from data.data import build_categorical_dataset
import matplotlib.pyplot as plt

def text_to_img(s : str, 
                generator : keras.models.Model, 
                char_map : Dict[str, int]) -> np.ndarray:
    words = s.lower().split(' ')
    images = []
    for word in words:
        noise = chars_to_noise(word, char_map, NOISE_IMG_DIM, N_CLASSES, seed=0)
        imgs = generator(noise)
        imgs = [tf.squeeze(imgs[i], -1) for i in range(imgs.shape[0])]
        images.append(imgs)
    
    image = vertical(images)
    return image

if __name__ == '__main__':
    _, char_map = build_categorical_dataset()
    char2idx = {v:k for k, v in char_map.items()}
    
    N_CLASSES = 27
    NOISE_IMG_DIM = 1
    generator = build_generator_model(NOISE_IMG_DIM, N_CLASSES-1)
    
    checkpoint_dir = './checkpoints/run41'
    checkpoint = tf.train.Checkpoint(
        generator=generator)
    
    try:
        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(checkpoint_dir, 'ckpts')))
    except Exception as e:
        print(f'ERROR: Failed to restore weights: {e}')
        exit(0)
        
    c = 'The quick brown fox jumps over the lazy dog'.lower()
    img = text_to_img(c, generator, char2idx)
    plt.imshow(img, cmap='gray')
    plt.show()