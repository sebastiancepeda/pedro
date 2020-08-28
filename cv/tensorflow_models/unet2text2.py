"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose,
    Dropout, Lambda, Input,
    MaxPooling2D, concatenate,
)


def identity_function(x):
    return x


def normalize_image_shape(height, width):
    base = (2 ** 4)
    width = (width // base) * base + base * int(width > 0)
    height = (height // base) * base + base * int(height > 0)
    return height, width


def get_model_definition(img_height, img_width, in_channels, out_channels):
    base = (2 ** 4)
    msg = "{actual} not multiple of " + str(base)
    assert img_height % base == 0, msg.format(actual=img_height)
    assert img_width % base == 0, msg.format(actual=img_width)
    inputs = Input((img_height, img_width, in_channels))
    x = Lambda(lambda aux: aux / 255)(inputs)
    # Downward
    kwargs_conv2d = {
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'padding': 'same',
    }
    side = 2
    k_size = (1+2*side,)*2
    h_dim = 100
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    # x = tf.keras.layers.Dense(100)(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(out_channels, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(out_channels, kernel_size=(1, 1), activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Softmax(axis=3)(x)
    print(outputs.shape)
    # Model compilation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    pre_process_input = identity_function
    return model, pre_process_input
