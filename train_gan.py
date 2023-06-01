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
from data.data import BATCH_SIZE, build_categorical_dataset
from utils import create_categorised_noise, images_to_gif
from utils import merge_metric_objects

N_CLASSES       = 27
NOISE_IMG_DIMS  = 1

@tf.function
def train_step(images, labels, n_classes):
    noise_data = [create_categorised_noise(NOISE_IMG_DIMS, n_classes) for _ in range(BATCH_SIZE)]
    noise = np.concatenate([n[0] for n in noise_data])
    noise_y = np.concatenate([n[1] for n in noise_data])
    
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_preds = discriminator(images, training=True)
        fake_preds = discriminator(generated_images, training=True)
        
        disc_loss = discriminator_loss(real_preds, labels, fake_preds, noise_y)
        
        # Accuracy if we want that, gotta update the functions to make them compatible uwu
        # disc_accuracy = discriminator_accuracy(real_preds, labels, fake_preds)
        # gen_accuracy = generator_accuracy(fake_preds)
        
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    
    with tf.GradientTape() as gen_tape:
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
    
def train(checkpoint_path, 
          checkpoint, 
          epochs, 
          data, 
          gen_seed, 
          gen_ys, 
          n_classes,
          char_map):
    checkpoint_save_path = os.path.join(checkpoint_path, 'ckpts', 'ckpt')
    images_file_path = os.path.join(checkpoint_path, 'imgs')
    metrics_file_path = os.path.join(checkpoint_path, 'metrics.json')
    data_cardinality = tf.data.experimental.cardinality(data)
    fig = plt.figure(figsize=(5, 7))
        
    if not os.path.exists(images_file_path):
        os.makedirs(images_file_path)
    
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
    # Load previous metrics 
    if os.path.exists(metrics_file_path):
        try:
            with open(metrics_file_path, 'r') as f:
                obj = json.load(f)
                metrics = merge_metric_objects(metrics, obj)
            print(f'Loaded previous metrics history')
        except Exception as e:
            print(f'Failed to merge dictionaries: {e}')
            exit(0)
            
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f'Epoch {epoch:2}/{epochs:}')
        
        epoch_metrics = {}
        for i, batch in enumerate(data):
            x, y = batch
            m = train_step(x, y, n_classes)

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
        
        fig_epoch = len(os.listdir(images_file_path)) + 1
        
        # Update figure with information
        plt.clf()
        fig.suptitle(f'Epoch {fig_epoch}')
        imgs = generator(gen_seed, training=False)
        for i in range(imgs.shape[0]):
            plt.subplot(6, 5, i+1)
            plt.imshow(imgs[i,:,:,0], cmap='gray')
            plt.title(f'{gen_ys[i]} ({char_map[gen_ys[i]]})')
            plt.axis('off')
        plt.pause(1)
        fig.savefig(os.path.join(images_file_path, f'epoch_{fig_epoch}.png'))
        images_to_gif(images_file_path, f'training.gif')
        
        # Update metrics
        metrics['generator']['loss'].append(float(gen_loss))
        metrics['generator']['accuracy'].append(float(gen_acc))
        metrics['discriminator']['loss'].append(float(disc_loss))
        metrics['discriminator']['accuracy'].append(float(disc_acc))
        
        # Save weights and metrics every nth epoch
        # We save weights and metrics at the same time to make the graph not crazy-looking
        if epoch % 5 == 0:
            print(f'Saved...', end=' ')
            checkpoint.save(file_prefix=checkpoint_save_path)
            with open(metrics_file_path, 'w') as f:
                json.dump(metrics, f)
        print()
    
    return metrics

if __name__ == '__main__':
    ds, char_map = build_categorical_dataset()

    generator = build_generator_model(NOISE_IMG_DIMS, N_CLASSES-1)
    discriminator = build_discriminator_model(N_CLASSES)
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
    
    checkpoint_dir = './checkpoints/run46'
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)
    
    try:
        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(checkpoint_dir, 'ckpts')))
    except:
        print(f'ERROR: Failed to restore weights.')
        exit(0)
        
    tf.random.set_seed(69420)
    gen_seed, gen_ys = create_categorised_noise(NOISE_IMG_DIMS, N_CLASSES, seed=0)
    
    epochs = 200
    metrics = train(
        checkpoint_dir, 
        checkpoint, 
        epochs, 
        ds, 
        gen_seed, 
        gen_ys, 
        N_CLASSES,
        char_map)
    
    plt.show(block=True)