from typing import *
import tensorflow as tf
import emnist

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024
SHUFFLE_SEED = 69420

TEST_SPLIT = 0.2

def reshape_data(x : tf.Tensor, y : tf.Tensor) -> tf.Tensor:
    x = tf.reshape(x, (28, 28, 1))
    return x, x

def build_dataset():
    x, y = emnist.extract_training_samples('balanced')
    
    x = x / 255.0
        
    data = (
        tf.data.Dataset.from_tensor_slices((x, x))
        .map(reshape_data)
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SHUFFLE_SEED)
        .batch(batch_size=BATCH_SIZE)
    )
    
    train_samples = round(tf.data.experimental.cardinality(data).numpy() * (1 - TEST_SPLIT))
    train_ds = data.take(train_samples)
    test_ds = data.skip(train_samples)
    
    train_samples = tf.data.experimental.cardinality(train_ds)
    test_samples = tf.data.experimental.cardinality(test_ds)
    print(f'Loaded {train_samples} Train Batches and {test_samples} Test Batches')
    
    return train_ds, test_ds