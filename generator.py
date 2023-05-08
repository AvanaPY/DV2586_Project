import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Dense, BatchNormalization, LeakyReLU, Reshape, Flatten


def build_generator_model():
    model = Sequential()
    model.add(Dense(7*7*32, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Reshape((7, 7, 32)))
    assert model.output_shape == (None, 7, 7, 32), 'Invalid output shape'
    
    model.add(Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', use_bias=False, activation='sigmoid'))
    return model

def build_discriminator_model():
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, 5, 2, 'same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, 5, 2, 'same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(16, 5, 2, 'same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    generator = build_generator_model()
    generator.summary()
    
    discriminator = build_discriminator_model()
    discriminator.summary()
    
    noise = tf.random.normal((1, 100))
    generated_image = generator(noise, training=False)
    decision = discriminator(generated_image)
    print(decision)
    
    plt.imshow(generated_image[0,:,:,0], cmap='gray')
    plt.show()