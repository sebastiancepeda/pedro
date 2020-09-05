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
    k_size = (11,)*2
    h_dim = 100
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    a = tf.keras.layers.Softmax(axis=1)(x)
    x = tf.concat([x, a * x], axis=3)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(out_channels, kernel_size=(1, 1), **kwargs_conv2d)(x)
    print(x.shape)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(481)(x)
    # x = tf.keras.layers.Reshape((-1, 1, 13, 37))(x)
    outputs = tf.keras.layers.Softmax(axis=3)(x)
    # Model compilation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    pre_process_input = identity_function
    return model, pre_process_input
