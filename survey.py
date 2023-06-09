from typing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

from data import build_categorical_dataset, create_pairs
from utils.noise import chars_to_noise
from model.generator import build_generator_model

import PIL
from PIL import Image

if __name__ == '__main__':
    _, char_map = build_categorical_dataset()

    generator = build_generator_model(1, 26)
    
    final_weights_path = 'model_weights/model2'
    checkpoint_dir = './checkpoints/run46'
    checkpoint = tf.train.Checkpoint(
        generator=generator)
    
    try:
        if not os.path.exists(os.path.join(final_weights_path, 'checkpoint')):
            raise FileNotFoundError(f'{final_weights_path} does not exist')
        checkpoint.restore(tf.train.latest_checkpoint(final_weights_path))
        print(f'Loaded up "{final_weights_path}"')
    except Exception as e1:
        try:
            checkpoint.restore(tf.train.latest_checkpoint(os.path.join(checkpoint_dir, 'ckpts')))
            checkpoint.save(os.path.join(final_weights_path, 'ckpt'))
            print(f'Loaded checkpoint, backing up to "{final_weights_path}"')
        except Exception as e2:
            print(f'ERROR: Failed to restore weights: {e2}')
            exit(0)
    
    pairs, classes = create_pairs(generator, 
                                  char_map, 
                                  n_class=26, 
                                  n_pairs=1)
    
    image_folder = 'survey_images'
    os.makedirs(image_folder, exist_ok=True)
    img_size = (128, 128)
    for ((real_image, gen_image), c) in zip(pairs, classes):
        c = char_map[c]
        
        real_image = Image.fromarray(real_image)
        real_image = real_image.resize(img_size, PIL.Image.NEAREST)
        real_image.save(os.path.join(image_folder, f'Real_{c}.png'))
        
        gen_image = Image.fromarray(gen_image).convert('L')
        gen_image = gen_image.resize(img_size, PIL.Image.NEAREST)
        gen_image.save(os.path.join(image_folder, f'Gen_{c}.png'))

    plt.figure(figsize=(10, 4))
    for i, ((real_image, generated_image), c) in enumerate(zip(pairs, classes)):
        plt.subplot(4, 13, i+1)
        plt.imshow(real_image, cmap='gray')
        plt.title(f'R {char_map[c]}')
        plt.axis('off')
        
        plt.subplot(4, 13, i+1+26)
        plt.imshow(generated_image, cmap='gray')
        plt.title(f'G {char_map[c]}')
        plt.axis('off')
    plt.savefig('alphabet.png')
    plt.show()