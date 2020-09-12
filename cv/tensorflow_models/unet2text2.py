"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, Lambda, Input,
    MaxPooling2D, )


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
    inputs = Input((img_height, img_width, in_channels))
    x = Lambda(lambda aux: aux / 255)(inputs)
    # Downward
    kwargs_conv2d = {
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'padding': 'same',
    }
    k_size = (3,) * 2
    h_dim = 10
    h1 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
    h1 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(h1)
    h2 = MaxPooling2D((2, 2))(h1)
    h2 = Conv2D(h_dim * 2, kernel_size=k_size, **kwargs_conv2d)(h2)
    h2 = Conv2D(h_dim * 2, kernel_size=k_size, **kwargs_conv2d)(h2)
    h3 = MaxPooling2D((2, 2))(h2)
    h3 = Conv2D(h_dim * 2, kernel_size=k_size, **kwargs_conv2d)(h3)
    h3 = Conv2D(h_dim * 2, kernel_size=k_size, **kwargs_conv2d)(h3)

    fg = compose_fs([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(out_channels * 2),
        tf.keras.layers.Dense(out_channels),
        tf.keras.layers.Softmax(axis=1),
        tf.keras.layers.Reshape((-1, 1, 1, 37)),
    ])

    outputs = []
    for it in range(13):
        # First glimpse
        a1 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(x)
        a1 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(a1)
        a1 = Conv2D(2, kernel_size=k_size, **kwargs_conv2d)(a1)
        a1 = tf.keras.layers.Softmax(axis=-1)(a1)
        a1 = a1[:, :, :, 0:1]
        a1 = tf.tile(a1, multiples=[1, 1, 1, h1.shape[-1]])
        g1 = tf.concat([h1, a1 * h1], axis=-1)
        # Second glimpse
        a2 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(g1)
        a2 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(a2)
        a2 = MaxPooling2D((2, 2))(a2)
        a2 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(a2)
        a2 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(a2)
        a2 = Conv2D(2, kernel_size=k_size, **kwargs_conv2d)(a2)
        a2 = tf.keras.layers.Softmax(axis=-1)(a2)
        a2 = a2[:, :, :, 0:1]
        a2 = tf.tile(a2, multiples=[1, 1, 1, h2.shape[-1]])
        g2 = tf.concat([h2, a2 * h2], axis=-1)
        # Third glimpse
        a3 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(g2)
        a3 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(a3)
        a3 = MaxPooling2D((2, 2))(a3)
        a3 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(a3)
        a3 = Conv2D(h_dim, kernel_size=k_size, **kwargs_conv2d)(a3)
        a3 = Conv2D(2, kernel_size=k_size, **kwargs_conv2d)(a3)
        a3 = tf.keras.layers.Softmax(axis=-1)(a3)
        a3 = a3[:, :, :, 0:1]
        a3 = tf.tile(a3, multiples=[1, 1, 1, h3.shape[-1]])
        g3 = tf.concat([h3, a3 * h3], axis=-1)
        # Flattening
        g = fg(g3)
        outputs.append(g)
    outputs = tf.concat(outputs, axis=3)
    outputs = tf.keras.layers.Reshape((-1, 13, 37))(outputs)
    print(outputs.shape)
    # Model compilation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    pre_process_input = identity_function
    return model, pre_process_input
