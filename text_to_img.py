import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from model.generator import build_generator_model, build_discriminator_model
from utils.noise import chars_to_noise, str_to_noise, empty_character
from data.data import build_categorical_dataset

import matplotlib.pyplot as plt

N_CLASSES = 27
NOISE_IMG_DIM = 4

if __name__ == '__main__':
    _, char_map = build_categorical_dataset()
    char2idx = {v:k for k, v in char_map.items()}
    generator = build_generator_model(NOISE_IMG_DIM, N_CLASSES-1)
    discriminator = build_discriminator_model(N_CLASSES)
    
    checkpoint_dir = './checkpoints/run32'
    checkpoint = tf.train.Checkpoint(
        generator=generator)
    
    try:
        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(checkpoint_dir, 'ckpts')))
    except:
        print(f'ERROR: Failed to restore weights.')
        exit(0)
        
    c = 'hello I like potato'.lower()
    noises = str_to_noise(c, char2idx, NOISE_IMG_DIM, N_CLASSES)
    imgs = []
    for noise in noises:
        img = generator(noise)
        for i in img:
            imgs.append(i)
        imgs.append(empty_character)
    
    for i, item in enumerate(imgs):
        plt.subplot(6, 6, i+1)
        plt.imshow(item, cmap='gray')
    plt.show()