from typing import *
import tensorflow as tf
import emnist
import string

N_CLASSES = 27
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024
SHUFFLE_SEED = 69420

def reshape_data_categorical(x : tf.Tensor, y : tf.Tensor) -> tf.Tensor:
    x = tf.reshape(x, (28, 28, 1))
    return x, y

def build_categorical_dataset():
    x, y = emnist.extract_training_samples('letters')

    x = x / 255.0
    
    idx2char = {i+1:c for i, c in enumerate(string.ascii_lowercase)}
    idx2char[0] = None
    
    char2idx = {v:k for k, v in idx2char.items()}
    
    char_map = {**idx2char, **char2idx}
    
    
    ds = (
        tf.data.Dataset.from_tensor_slices((x, y))
        .map(reshape_data_categorical)
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SHUFFLE_SEED)
        .batch(batch_size=(N_CLASSES-1)*BATCH_SIZE, drop_remainder=True)
    )
    
    return ds, char_map