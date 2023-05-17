import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as ktuner

from data.data import build_dataset
from model.model import build_autoencoder, load_autoencoder, save_autoencoder

filepath = 'models/emnist_ae'

train_ds, test_ds = build_dataset()
# ae = load_autoencoder(filepath=filepath)

hp = ktuner.HyperParameters()
tuner = ktuner.RandomSearch(
    hypermodel=build_autoencoder,
    objective='val_loss',
    max_trials=8,
    executions_per_trial=2,
    directory='hypertuning',
    project_name='LatentDropoutDeeperDecoder1'
)

tuner.search_space_summary()
tuner.search(
    train_ds.take(200), 
    validation_data=test_ds.take(50), 
    epochs=4
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
ae = tuner.hypermodel.build(best_hps)
    
print(f'Built model with parameters:')
for param_name, param_value in best_hps.values.items():
    if isinstance(param_value, int):
        print(f'{param_name.rjust(10)}: {param_value:2,}')
    elif isinstance(param_value, float):
        print(f'{param_name.rjust(10)}: {param_value:10.7f}')

plt.figure(figsize=(16, 8))

display_ds = train_ds.take(1)
for batch in display_ds:
    xs, ys = batch
    imgs = ae(xs)
    
    for i in range(1,4):
        ax = plt.subplot(2, 6, i)
        ax2 = plt.subplot(2, 6, i+6)
        
        ax.imshow(xs[i-1], cmap='gray')
        ax2.imshow(imgs[i-1], cmap='gray')

ae.fit(train_ds, validation_data=test_ds, epochs=10)
save_autoencoder(ae, filepath=filepath)

for batch in display_ds:
    xs, ys = batch
    imgs = ae(xs)
    
    for i in range(1, 4):
        ax = plt.subplot(2, 6, i+3)
        ax2 = plt.subplot(2, 6, i+6+3)
        
        ax.imshow(xs[i-1], cmap='gray')
        ax2.imshow(imgs[i-1], cmap='gray')
    
plt.show()