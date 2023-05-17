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
    
    model.add(Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same', use_bias=False, activation='sigmoid'))
    
    assert model.output_shape == (None, 28, 28, 1)
    return model

def build_discriminator_model():
    model = Sequential()
    model.add(Conv2D(filters=218, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=32, kernel_size=5, strides=2, padding='same'))
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

def discriminator_accuracy(real, fake):
    real_acc = tf.keras.metrics.binary_accuracy(tf.ones_like(real), real)
    fake_acc = tf.keras.metrics.binary_accuracy(tf.zeros_like(fake), fake)
    
    f_real_acc = tf.math.reduce_mean(real_acc)
    f_fake_acc = tf.math.reduce_mean(fake_acc)
    return (f_real_acc + f_fake_acc) / 2

def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

def generator_accuracy(fake_output):
    return tf.keras.metrics.binary_accuracy(tf.ones_like(fake_output), fake_output)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    generator = build_generator_model()
    generator.summary()
    
    discriminator = build_discriminator_model()
    discriminator.summary()
    
    try:
        generator_optimizer = tf.keras.optimizers.RMSprop(1e-3)
        discriminator_optimizer = tf.keras.optimizers.Adam(3e-4)
        checkpoint_dir = './checkpoints/run2'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
        tf.random.set_seed(69420)
        noise = tf.random.normal((25, 100))
        generated_image = generator(noise, training=False)
        decision = discriminator(generated_image)
        
        plt.figure(figsize=(12, 12))
        for i in range(noise.shape[0]):
            plt.subplot(5, 5, i+1)
            plt.imshow(generated_image[i,:,:,0], cmap='gray')
        plt.show()
    except Exception as e:
        print(f'Could not restore model.')