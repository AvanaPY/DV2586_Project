import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model.generator import build_discriminator_model, build_generator_model
from model.generator import discriminator_loss, discriminator_accuracy, generator_loss, generator_accuracy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data.data import build_dataset, BATCH_SIZE, build_categorical_dataset

NOISE_DIM = 100

plt.ion()

@tf.function
def train_step(images, labels, n_classes, train_model_indicator : int):
    noise_data = [create_categorised_noise(7, n_classes) for _ in range(BATCH_SIZE)]
    noise = np.concatenate([n[0] for n in noise_data])
    noise_y = np.concatenate([n[1] for n in noise_data])
    # noise, noise_y = create_categorised_noise(7, n_classes)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_preds = discriminator(images, training=True)
        fake_preds = discriminator(generated_images, training=True)
        
        disc_loss = discriminator_loss(real_preds, labels, fake_preds, noise_y)
        gen_loss = generator_loss(fake_preds, noise_y)
        
        # disc_accuracy = discriminator_accuracy(real_preds, labels, fake_preds)
        # gen_accuracy = generator_accuracy(fake_preds)
        
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    # If we wish to train the generator and discriminator separately, one can use these lines of code
    # if train_model_indicator == 0:
    #     generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    # else:
    #     discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    gen_accuracy = 0
    disc_accuracy = 0
    ret = [
        [tf.math.reduce_mean(gen_loss), tf.math.reduce_mean(gen_accuracy)], 
        [tf.math.reduce_mean(disc_loss), tf.math.reduce_mean(disc_accuracy)]
    ]
    return ret
    
def train(train_step, 
          heckpoint_prefix, 
          checkpoint, 
          epochs, 
          data, 
          gen_seed, 
          gen_ys, 
          n_classes,
          char_map):
    def get_key_create_not_exists(d : dict, key : any):
        if d.get(key, None) == None:
            d[key] = []
        return d.get(key)
    
    def join_metrics(metrics, new_metrics):
        gen_metrics, disc_metrics = new_metrics
        
        get_key_create_not_exists(metrics, 'gen_loss').append(gen_metrics[0])
        get_key_create_not_exists(metrics, 'gen_acc').append(gen_metrics[1])
        get_key_create_not_exists(metrics, 'disc_loss').append(disc_metrics[0])
        get_key_create_not_exists(metrics, 'disc_acc').append(disc_metrics[1])
        
    train_gen_indicator = tf.Variable(0)
    train_disc_indicator = tf.Variable(1)
    data_cardinality = tf.data.experimental.cardinality(data)
    
    fig = plt.figure(figsize=(10, 12))
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch:2}/{epochs:}')
        
        metrics = {}
        for i, batch in enumerate(data):
            x, y = batch

            f_turn = train_gen_indicator if epoch % 2 == 0 else train_disc_indicator
            s_turn = train_gen_indicator if epoch % 2 == 1 else train_disc_indicator
            if (i // 25) % 2 == 0:
                m = train_step(x, y, n_classes, f_turn)
            else:
                m = train_step(x, y, n_classes, s_turn)

            join_metrics(metrics, m)
            
            bar_length = 50
            progress = (i+1) / data_cardinality
            progress_ticks = int(bar_length * progress)
            
            gen_loss = np.mean(metrics['gen_loss'])
            gen_acc = np.mean(metrics['gen_acc'])
            disc_loss = np.mean(metrics['disc_loss'])
            disc_acc = np.mean(metrics['disc_acc'])
            
            print(f'\r{i+1:3} / {data_cardinality}[{"="*progress_ticks}>{" "*(bar_length-progress_ticks)}] Generator[Loss: {gen_loss:8.4f}, Accuracy:{gen_acc:6.4f}], Discriminator[Loss: {disc_loss:8.4f}, Accuracy:{disc_acc:6.4f}]', end='')
        
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print()
        
        save_fig_dir = os.path.join(checkpoint_prefix, 'a_imgs')
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)
        
        fig_epoch = len(os.listdir(save_fig_dir)) + 1
        
        # Update figure with information
        plt.clf()
        fig.suptitle(f'Epoch {fig_epoch}')
        imgs = generator(gen_seed, training=False)
        for i in range(imgs.shape[0]):
            plt.subplot(6, 6, i+1)
            plt.imshow(imgs[i,:,:,0], cmap='gray')
            plt.title(f'{gen_ys[i]} ({char_map[gen_ys[i]]})')
            plt.axis('off')
        plt.pause(1)
        fig.savefig(os.path.join(save_fig_dir, f'epoch_{fig_epoch}.png'))

def create_categorised_noise(noise_img_dim : int, n_classes : int):
    n_classes -= 1
    noise_imgs = np.zeros(shape=(n_classes, noise_img_dim, noise_img_dim, n_classes))
    noise_y = np.arange(n_classes) + 1
    for i in range(n_classes):
        noise_imgs[i,:,:,i] = np.random.uniform(low=0, high=1,size=(noise_img_dim, noise_img_dim))
    return noise_imgs, noise_y

if __name__ == '__main__':
    ds, char_map = build_categorical_dataset()
    n_classes = len(char_map.keys())
    noise_imgs, noise_y = create_categorised_noise(7, n_classes)
    
    generator = build_generator_model(7, n_classes)
    discriminator = build_discriminator_model(n_classes)
    
    generator.summary()
    discriminator.summary()
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
    
    checkpoint_dir = './checkpoints/run14'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    tf.random.set_seed(69420)
    gen_seed, gen_ys = create_categorised_noise(7, n_classes)
    
    epochs = 100
    train(train_step, 
          checkpoint_prefix, 
          checkpoint, 
          epochs, 
          ds, 
          gen_seed, 
          gen_ys, 
          n_classes,
          char_map)
    
    imgs = generator(gen_seed, training=False)
    plt.clf()
    for i in range(imgs.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i,:,:,0], cmap='gray')
    
    plt.show(block=True)