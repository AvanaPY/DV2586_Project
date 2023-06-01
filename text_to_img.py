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
from PIL import Image

def text_to_img(s : str, 
                generator : keras.models.Model, 
                char_map : Dict[str, int],
                transparent : bool = False,
                transparency_threshold : int = 50) -> np.ndarray:
    segments = s.lower().split('\n')
    images = []
    for i, segment in enumerate(segments):
        row = []
        words = segment.split(' ')
        for word_i, word in enumerate(words):
            noise = chars_to_noise(word, char_map, NOISE_IMG_DIM, N_CLASSES, seed=0)
            imgs = generator(noise)
            imgs = [tf.squeeze(imgs[i], -1) for i in range(imgs.shape[0])]
            row.extend(imgs)
            if word_i < len(words) - 1:
                row.append(np.zeros((28, 28)))
        images.append(row)

    img = vertical(images, center=True)
    img *= 255
    img = np.round(img).astype(np.uint8)
    
    image = Image.fromarray(img)
    
    if transparent:
        image = image.convert('RGBA')
        newData = []
        
        for item in image.getdata():
            gs = sum(item[0:3]) / 3
            if gs < transparency_threshold:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        image.putdata(newData)
        
    return image

if __name__ == '__main__':
    _, char_map = build_categorical_dataset()
    char2idx = {v:k for k, v in char_map.items()}
    
    N_CLASSES = 27
    NOISE_IMG_DIM = 1
    generator = build_generator_model(NOISE_IMG_DIM, N_CLASSES-1)
    
    checkpoint_dir = './checkpoints/run46'
    checkpoint = tf.train.Checkpoint(
        generator=generator)
    
    try:
        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(checkpoint_dir, 'ckpts')))
    except Exception as e:
        print(f'ERROR: Failed to restore weights: {e}')
        exit(0)
        
    # c = 'The quick brown fox jumps over the lazy dog'
    c = 'gan it write'
    image = text_to_img(c, generator, char2idx, transparent=True)
    image.save('generated_image.png')
    
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()