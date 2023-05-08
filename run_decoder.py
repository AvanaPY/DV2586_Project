import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from model import load_autoencoder
from data import build_dataset

def update_plot():
    out = ae.decoder(data)
    axes_image.set_data(out[0])

def on_slider_change_wrapper(slider : int):
    def wrapped(f : float):
        data[0][slider] = f
        update_plot()
    return wrapped


LATENT_DIMS = 10
MODEL_NAME = 'models/emnist_ae'

ae = load_autoencoder(MODEL_NAME)
ae.decoder.summary()
    
data, _ = build_dataset()

for batch in data.take(1):
    x, _ = batch
    data = tf.reshape(ae.encoder(x)[90], (1, LATENT_DIMS)).numpy()
    
# data = (np.random.rand(1, LATENT_DIMS) - 0.5) * 0.5
out = ae.decoder(data)

fig = plt.figure(figsize=(12, 12))
for i in range(len(data)):
    ax = plt.subplot(1, 2, i+2)

    img = out[i]
    axes_image = ax.imshow(img, cmap='gray')

x = 0.1
y = 0.98
w = 0.35
h = 0.01
STEP = 0.02

sliders = []
for feature_id in range(LATENT_DIMS):

    slider_ax = fig.add_axes([x, y - feature_id * STEP, w, h])
    sliders.append(Slider(
        ax=slider_ax,
        label=f'Feature {feature_id+1}',
        valmin=-1,
        valmax=1,
        valinit=0,
        orientation='horizontal'
    ))

for i, slider in enumerate(sliders):
    slider.set_val(data[0][i])
    slider.on_changed(on_slider_change_wrapper(i))

plt.show()