from typing import *
import tensorflow as tf
import emnist
import string

N_CLASSES = 27
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024
SHUFFLE_SEED = 69420

TEST_SPLIT = 0.2

def reshape_data(x : tf.Tensor, y : tf.Tensor) -> tf.Tensor:
    x = tf.reshape(x, (28, 28, 1))
    return x, x

def reshape_data_categorical(x : tf.Tensor, y : tf.Tensor) -> tf.Tensor:
    x = tf.reshape(x, (28, 28, 1))
    return x, y

def build_dataset(no_validation:bool = False):
    x, y = emnist.extract_training_samples('balanced')
    
    x = x / 255.0
        
    data = (
        tf.data.Dataset.from_tensor_slices((x, x))
        .map(reshape_data)
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SHUFFLE_SEED)
        .batch(batch_size=BATCH_SIZE, drop_remainder=True)
    )
    
    if no_validation:
        print(f'Loaded {tf.data.experimental.cardinality(data)} data batches and no validation data.')
        return data
    
    train_samples = round(tf.data.experimental.cardinality(data).numpy() * (1 - TEST_SPLIT))
    train_ds = data.take(train_samples)
    test_ds = data.skip(train_samples)
    
    train_samples = tf.data.experimental.cardinality(train_ds)
    test_samples = tf.data.experimental.cardinality(test_ds)
    print(f'Loaded {train_samples} Train Batches and {test_samples} Test Batches')
    
    return train_ds, test_ds

def build_categorical_dataset():
    x, y = emnist.extract_training_samples('letters')

    x = x / 255.0
    
    character_mapping = {i+1:c for i, c in enumerate(string.ascii_lowercase)}
    character_mapping[0] = None
    
    ds = (
        tf.data.Dataset.from_tensor_slices((x, y))
        .map(reshape_data_categorical)
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SHUFFLE_SEED)
        .batch(batch_size=(N_CLASSES-1)*BATCH_SIZE, drop_remainder=True)
    )
    
    return ds, character_mapping