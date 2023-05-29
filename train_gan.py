import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model.generator import build_discriminator_model, build_generator_model
from model.generator import discriminator_loss, discriminator_accuracy, generator_loss, generator_accuracy
import tensorflow as tf
import numpy as np
import time
import json
import matplotlib.pyplot as plt
plt.ion()
from data.data import build_dataset, BATCH_SIZE, build_categorical_dataset
from utils import create_categorised_noise, images_to_gif
from utils import merge_metric_objects

N_CLASSES       = 27
NOISE_IMG_DIMS  = 4

@tf.function
def train_step(images, labels, n_classes, train_model_indicator : int):
    noise_data = [create_categorised_noise(NOISE_IMG_DIMS, n_classes) for _ in range(BATCH_SIZE)]
    noise = np.concatenate([n[0] for n in noise_data])
    noise_y = np.concatenate([n[1] for n in noise_data])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_preds = discriminator(images, training=True)
        fake_preds = discriminator(generated_images, training=True)
        
        disc_loss = discriminator_loss(real_preds, labels, fake_preds, noise_y)
        
        # Accuracy if we want that, gotta update the functions to make them compatible uwu
        # disc_accuracy = discriminator_accuracy(real_preds, labels, fake_preds)
        # gen_accuracy = generator_accuracy(fake_preds)
        
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        fake_preds = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_preds, noise_y)
        
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    
    gen_accuracy = 0
    disc_accuracy = 0
    ret = [
        [tf.math.reduce_mean(gen_loss), tf.math.reduce_mean(gen_accuracy)], 
        [tf.math.reduce_mean(disc_loss), tf.math.reduce_mean(disc_accuracy)]
    ]
    return ret
    
def train(train_step, 
          checkpoint_path, 
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
    
    fig = plt.figure(figsize=(5, 6))
    metrics = {
        'generator' : {
            'loss': [],
            'accuracy' : []
        },
        'discriminator' : {
            'loss': [],
            'accuracy' : []
        }
    }
    checkpoint_save_path = os.path.join(checkpoint_dir, 'ckpts', 'ckpt')
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f'Epoch {epoch:2}/{epochs:}')
        
        epoch_metrics = {}
        for i, batch in enumerate(data):
            x, y = batch

            f_turn = train_gen_indicator if epoch % 2 == 0 else train_disc_indicator
            s_turn = train_gen_indicator if epoch % 2 == 1 else train_disc_indicator
            if (i // 25) % 2 == 0:
                m = train_step(x, y, n_classes, f_turn)
            else:
                m = train_step(x, y, n_classes, s_turn)

            join_metrics(epoch_metrics, m)
            
            bar_length = 50
            progress = (i+1) / data_cardinality
            progress_ticks = int(bar_length * progress)
            
            gen_loss = np.mean(epoch_metrics['gen_loss'])
            gen_acc = np.mean(epoch_metrics['gen_acc'])
            disc_loss = np.mean(epoch_metrics['disc_loss'])
            disc_acc = np.mean(epoch_metrics['disc_acc'])
            
            epoch_time = time.time() - epoch_start
            print(f'\r{i+1:3} / {data_cardinality}[{"="*progress_ticks}>{" "*(bar_length-progress_ticks)}] {epoch_time:.0f}s Generator[Loss: {gen_loss:8.4f}, Accuracy:{gen_acc:6.4f}], Discriminator[Loss: {disc_loss:8.4f}, Accuracy:{disc_acc:6.4f}]', end=' ')
        
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_save_path)
            print(f'Saved weights...', end=' ')
        print()
        
        save_fig_dir = os.path.join(checkpoint_path, 'imgs')
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
        images_to_gif(save_fig_dir, f'training.gif')
        
        metrics['generator']['loss'].append(float(gen_loss))
        metrics['generator']['accuracy'].append(float(gen_acc))
        metrics['discriminator']['loss'].append(float(disc_loss))
        metrics['discriminator']['accuracy'].append(float(disc_acc))

    return metrics

if __name__ == '__main__':
    ds, char_map = build_categorical_dataset()
    
    noise_imgs, noise_y = create_categorised_noise(NOISE_IMG_DIMS, N_CLASSES)
    
    generator = build_generator_model(NOISE_IMG_DIMS, N_CLASSES-1)
    discriminator = build_discriminator_model(N_CLASSES)
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    checkpoint_dir = './checkpoints/run32'
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)
    
    try:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    except:
        print(f'ERROR: Failed to restore weights.')
        exit(0)
        
    tf.random.set_seed(69420)
    gen_seed, gen_ys = create_categorised_noise(NOISE_IMG_DIMS, N_CLASSES)
    
    epochs = 50
    metrics = train(train_step, 
        checkpoint_dir, 
        checkpoint, 
        epochs, 
        ds, 
        gen_seed, 
        gen_ys, 
        N_CLASSES,
        char_map)
    # Load previous metrics 
    try:
        json_file_path = os.path.join(checkpoint_dir, 'metrics.json')
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                obj = json.load(f)
                metrics = merge_metric_objects(metrics, obj)
    except Exception as e:
        print(f'Failed to merge dictionaries: {e}')
            
    with open(json_file_path, 'w') as f:
        json.dump(metrics, f)
    
    imgs = generator(gen_seed, training=False)
    plt.clf()
    for i in range(imgs.shape[0]):
        plt.subplot(6, 6, i+1)
        plt.imshow(imgs[i,:,:,0], cmap='gray')
    
    plt.show(block=True)