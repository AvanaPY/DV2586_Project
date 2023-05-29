from __future__ import annotations
from typing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import Model, Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, UpSampling2D, Reshape, Dropout
import keras_tuner as kt

class AutoEncoder(Model):
    def __init__(self, encoder : Sequential, decoder : Sequential, full : Model):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._full = full
    
    @property
    def model(self):
        return self._full
    
    @property
    def encoder(self):
        return self._encoder
    
    @property
    def decoder(self):
        return self._decoder
    
    def call(self, *args, **kwargs) -> Any:
        return self._full(*args, **kwargs)
    
    def summary(self):
        self._full.summary()
        
    def fit(self, *args, **kwargs):
        return self._full.fit(*args, **kwargs)
    
    def save(self, filepath : str) -> None:
        self._decoder.save(os.path.join(filepath, 'decoder.m'))     
        self._encoder.save(os.path.join(filepath, 'encoder.m'))   
        
    @staticmethod
    def load(filepath : str) -> AutoEncoder:
        encoder = tf.keras.models.load_model(os.path.join(filepath, 'encoder.m'))
        decoder = tf.keras.models.load_model(os.path.join(filepath, 'decoder.m'))
            
        inp = Input(shape=(28, 28, 1))
        enc = encoder(inp)
        out = decoder(enc)
        
        model = Model(inputs=inp, outputs=[out])
        model.build((None, 28, 28, 1))
        
        return AutoEncoder(encoder, decoder, model)

def load_autoencoder(filepath : str) -> AutoEncoder:
    return AutoEncoder.load(filepath)

def save_autoencoder(ae : AutoEncoder, filepath : str) -> None:
    return ae.save(filepath)

def build_autoencoder(hp : kt.HyperParameters) -> AutoEncoder:
    
    # Define hyper parameters
    LATENT_DIMS = 10
    LEARNING_RATE = hp.Float('lr', min_value=1e-5, max_value=1e-3, sampling='log')
    DROPOUT_RATE  = hp.Float('dropout', min_value=0, max_value=0.2, step=0.04)

    # Build model
    encoder = Sequential(name='Encoder')
    encoder.add(Conv2D(128, kernel_size=5, strides=1, padding='valid', activation='relu', name='EncoderStart'))
    encoder.add(Conv2D(64, kernel_size=5, strides=1, padding='valid', activation='relu'))
    encoder.add(BatchNormalization(momentum=0.8))
    encoder.add(Conv2D(32, kernel_size=5, strides=1, padding='valid', activation='relu'))
    encoder.add(BatchNormalization(momentum=0.8))
    encoder.add(Conv2D(1, kernel_size=5, strides=1, padding='valid', activation='relu'))
    encoder.add(Flatten())
    encoder.add(Dense(LATENT_DIMS, activation='tanh'))
    encoder.add(Dropout(DROPOUT_RATE))
    # Decoder
    decoder = Sequential(name='Decoder')
    decoder.add(Dense(100, activation='relu', name='DecoderStart'))
    decoder.add(Reshape(target_shape=(10, 10, 1)))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu'))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu'))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu'))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu'))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu'))
    decoder.add(Conv2D(filters=1, kernel_size=1, strides=1, padding='valid', activation='sigmoid'))
    
    # Model
    ae = final_encoder(LEARNING_RATE, encoder, decoder)
    
    return ae

def final_encoder(LEARNING_RATE : kt.HyperParameter, encoder : Model, decoder : Model) -> AutoEncoder:
    inp = Input(shape=(28, 28, 1))
    enc = encoder(inp)
    out = decoder(enc)
    
    model = Model(inputs=inp, outputs=[out])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [] # For the love of god future Emil, don't use accuracy like you have done twice (Do it //Sam)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    model.build(input_shape=(None, 28, 28, 1))
    
    ae = AutoEncoder(encoder, decoder, model)
    return ae

if __name__ == '__main__':
    hp = kt.HyperParameters()
    autoencoder = build_autoencoder(hp)
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()