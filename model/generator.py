import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Dense, BatchNormalization, LeakyReLU, Reshape, Flatten

class ResidualConv2D(keras.layers.Layer):
    def __init__(self, filters : int, kernel_size : int):
        super().__init__()
        self._conv2d1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')
        self._bn = BatchNormalization()
        self._lrelu = LeakyReLU()
        
    def call(self, input):
        x = self._conv2d1(input)
        x = self._bn(x)
        x = self._lrelu(x)
        x = x + input
        return x
        
def build_generator_model(noise_img_dim : int, n_classes : int):
    model = Sequential(name='StureGAN_Generator') 
    model.add(Flatten())
    model.add(Dense(7*7*8))
    model.add(LeakyReLU())
    
    model.add(Reshape((7, 7, -1)))
    
    model.add(Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=256, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid'))
    
    model.build((None, n_classes, noise_img_dim))
    return model

def build_discriminator_model(n_classes : int):
    model = Sequential(name='StureGAN_Discriminator')
    model.add(Conv2D(filters=128, kernel_size=7, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    
    model.add(ResidualConv2D(128, 3))
    model.add(ResidualConv2D(128, 3))
    model.add(ResidualConv2D(128, 3))
    
    model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU())
    
    model.add(ResidualConv2D(64, 5))
    model.add(ResidualConv2D(64, 5))
    model.add(ResidualConv2D(64, 5))
    
    model.add(Conv2D(filters=64, kernel_size=1, strides=1, padding='same'))
    
    # Run41
    # model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
    # model.add(LeakyReLU())
    
    # model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
    # model.add(LeakyReLU())
    
    # model.add(Conv2D(filters=64, kernel_size=1, strides=1, padding='same'))
    # model.add(LeakyReLU())
    
    model.add(Flatten())
    
    model.add(Dense(n_classes, activation='softmax'))
    return model

def discriminator_loss(real_preds, real_y, fake_preds, fake_y):
    real_loss = tf.keras.losses.sparse_categorical_crossentropy(real_y, real_preds)
    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros_like(fake_y), fake_preds)
    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss

def discriminator_accuracy(real, fake):
    real_acc = tf.keras.metrics.binary_accuracy(tf.ones_like(real), real)
    fake_acc = tf.keras.metrics.binary_accuracy(tf.zeros_like(fake), fake)
    
    f_real_acc = tf.math.reduce_mean(real_acc)
    f_fake_acc = tf.math.reduce_mean(fake_acc)
    return (f_real_acc + f_fake_acc) / 2

def generator_loss(fake_output, fake_y):
    return tf.keras.losses.sparse_categorical_crossentropy(fake_y, fake_output)

def generator_accuracy(fake_output):
    return tf.keras.metrics.binary_accuracy(tf.ones_like(fake_output), fake_output)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    generator = build_generator_model(4, 26)
    generator.summary()
    
    discriminator = build_discriminator_model(27)
    discriminator.summary()