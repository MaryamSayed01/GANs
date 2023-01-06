# import cv2
import numpy as np
import pandas as pd
import glob
import os
import time
# import cv2
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
# %matplotlib inline
from tensorflow import keras
import keras
from keras.layers import Layer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose,add,Lambda, ReLU,Input,Layer,Conv2D,Reshape , concatenate,Dropout,Lambda,Conv2DTranspose, multiply,MaxPool2D, LeakyReLU,Concatenate,UpSampling2D,Conv2DTranspose,BatchNormalization,MaxPooling2D,Input
from keras.models import Model, load_model
from keras import backend as K
# import Path
from PIL import Image
from IPython.core.display import Path
from tensorflow.keras import Input, Model
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import io
from PIL import Image
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
input_img=(128,128,3)
latent_dim = 10
enc_weights = np.load('enc_weights.npy', allow_pickle=True)
dec_weights = np.load('dec_weights.npy', allow_pickle=True)
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
def encoder(input_encoder):
    inputs = keras.Input(shape=input_encoder, name='i')
    # Block-1
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(inputs)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Block-2
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Block-3
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Block-4
    x = Conv2D(64, 3, 2, padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Final Block
    flatten = Flatten()(x)
    mean = Dense(10, name='mean')(flatten)
    log_var = Dense(10, name='siqma')(flatten)
    # out = Lambda(sampling_reparameterization,name='Z')([mean, log_var])
    z = Sampling()([mean, log_var])
    model = tf.keras.Model(inputs, [mean, log_var, z])
    return model
def decoder(input_decoder):
    inputs = keras.Input(shape=input_decoder)
    x = Dense(65536)(inputs)
    x = Reshape((32, 32, 64))(x)
    # Block-1
    x = Conv2DTranspose(64, 3, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Block-2
    x = Conv2DTranspose(64, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Block-3
    x = Conv2DTranspose(32, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Block-4
    outputs = Conv2DTranspose(3, 3, 1, padding='same', activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model
st.set_page_config(page_title = 'Variational Autoencoder',page_icon = '🎗️')
enc=encoder(input_img)
dec=decoder(latent_dim)
enc.set_weights(enc_weights[61])
dec.set_weights(dec_weights[61])

def main():
    st.title("Variational Autoencoder")
    st.sidebar.title("Uploaded image")
    st.markdown(" Welcome To Our VAE System")
    file_upload = st.file_uploader("upload a file image", type=['jpeg', 'jpg', 'png'], accept_multiple_files=True)
    images,generated_images=[],[]
    i=0
    if file_upload:
        for img_file in file_upload:
            file = img_file.read()
            img = Image.open(img_file)
            st.sidebar.image(img, caption='Input', use_column_width=True)
            img = img.resize(size=(128, 128))
            img = np.array(img)/255.
            m,v,z=enc(np.array([img]))
            generated_img=dec(z)
            generated_img=np.array(generated_img)
            generated_images.append(generated_img)
            st.image(generated_img, caption='output', use_column_width=True)
if __name__ == '__main__':
    main()
