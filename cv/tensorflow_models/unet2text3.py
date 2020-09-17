"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, Lambda, Input,
    MaxPooling2D)
import numpy as np


def identity_function(x):
    return x


def normalize_image_shape(height, width):
    base = (2 ** 4)
    width = (width // base) * base + base * int(width > 0)
    height = (height // base) * base + base * int(height > 0)
    return height, width


def compose_fs(functions):
    def composed_function(x):
        for f in functions:
            x = f(x)
        return x

    return composed_function


def get_model_definition(img_height, img_width, in_channels, out_channels):
    base = (2 ** 4)
    msg = "{actual} not multiple of " + str(base)
    assert img_height % base == 0, msg.format(actual=img_height)
    assert img_width % base == 0, msg.format(actual=img_width)
    input_image = Input((img_height, img_width, in_channels))
    x = Lambda(lambda aux: aux / 255)(input_image)
    # Downward
    kwargs_conv2d = {
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'padding': 'same',
    }
    timesteps = 13
    abecedary_len = 37
    k_size = (3,) * 2
    h_dim = 50

    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h)
    h = MaxPooling2D((2, 2))(h)
    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h)
    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h)
    h = MaxPooling2D((2, 2))(h)
    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h)
    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h)
    h = MaxPooling2D((2, 2))(h)
    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h)
    h = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h)

    def position_f(pos):
        pos = np.identity(abecedary_len)[pos]
        pos = np.reshape(pos, (-1, abecedary_len))
        pos = tf.constant(pos, dtype='float32')
        return pos

    x2_f = compose_fs((
        Conv2D(2, kernel_size=k_size, **kwargs_conv2d),
        tf.keras.layers.Flatten(),
    ))
    attention_f = compose_fs((
        tf.keras.layers.Dense(h.shape[1] * h.shape[2] * 2),
        tf.keras.layers.Dense(h.shape[1] * h.shape[2] * 2),
        tf.keras.layers.Dense(h.shape[1] * h.shape[2] * 2),
        tf.keras.layers.Reshape((h.shape[1], h.shape[2], 2)),
        tf.keras.layers.Softmax(axis=-1),
        lambda aux: aux[:, :, :, 0:1],
        lambda aux: tf.tile(aux, multiples=[1, 1, 1, h.shape[-1]])
    ))
    glimpse_f = compose_fs((
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(out_channels),
        tf.keras.layers.Softmax(axis=1),
        tf.keras.layers.Reshape((-1, 1, 1, abecedary_len)),
    ))
    outputs = []
    for it in range(timesteps):
        position = position_f(it)
        x2 = x2_f(h)
        x2 = tf.concat([x2, position], axis=-1)
        attention = attention_f(x2)
        glimpse = tf.concat([h, attention * h], axis=-1)
        glimpse = glimpse_f(glimpse)
        outputs.append(glimpse)
    outputs = tf.concat(outputs, axis=3)
    outputs = tf.keras.layers.Reshape((-1, timesteps, abecedary_len))(outputs)
    print(outputs.shape)
    # Model compilation
    model = Model(inputs=[input_image], outputs=[outputs])
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
    )
    pre_process_input = identity_function
    return model, pre_process_input
