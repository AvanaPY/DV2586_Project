import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from generator import build_discriminator_model, build_generator_model, discriminator_loss, generator_loss
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import build_dataset, BATCH_SIZE

NOISE_DIM = 100

@tf.function
def train_step(images, train_model_indicator : int):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real = discriminator(images, training=True)
        fake = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake)
        disc_loss = discriminator_loss(real, fake)
        
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    if train_model_indicator == 0:
        generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    else:
        discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    return tf.math.reduce_mean(gen_loss), tf.math.reduce_mean(disc_loss)
    
if __name__ == '__main__':
    generator = build_generator_model()
    discriminator = build_discriminator_model()
    
    generator.summary()
    discriminator.summary()
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)
    
    data, _ = build_dataset()
    
    checkpoint_dir = './checkpoints/proj6'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)
    
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    tf.random.set_seed(69420)
    
    train_gen_indicator = tf.Variable(0)
    train_disc_indicator = tf.Variable(1)
    epochs = 0
    data_cardinality = tf.data.experimental.cardinality(data)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1:2}/{epochs:}')
        print(f'{train_gen_indicator=} : 0:{train_gen_indicator==0}')
        print(f'{train_disc_indicator=}: 1:{train_disc_indicator==1}')
        
        
        epoch_gen_losses = []
        epoch_disc_losses = []
        for i, batch in enumerate(data):
            x, _ = batch
            if (i // 25) % 2 == 0:
                gen_loss, disc_loss = train_step(x, train_gen_indicator)
            else:
                gen_loss, disc_loss = train_step(x, train_disc_indicator)

            epoch_gen_losses.append(gen_loss)
            epoch_disc_losses.append(disc_loss)
            
            bar_length = 50
            progress = (i+1) / data_cardinality
            progress_ticks = int(bar_length * progress)
            
            gen_loss = np.mean(epoch_gen_losses)
            disc_loss = np.mean(epoch_disc_losses)
            print(f'\r{i:3}/ {data_cardinality}[{"="*progress_ticks}>{" "*(bar_length-progress_ticks)}] Loss: Generator: {gen_loss:6.4f} Discriminator: {disc_loss:6.4f}', end='')
        
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print()
    
    gen_seed = tf.random.normal([25, NOISE_DIM])
    imgs = generator(gen_seed)
    
    fig = plt.figure(figsize=(12, 12))
    for i in range(imgs.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i,:,:,0], cmap='gray')
    
    plt.show()