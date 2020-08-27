"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Lambda,
    Input,
    # AveragePooling2D,
    # concatenate,
)


def identity_function(x):
    return x


def normalize_image_shape(height, width):
    base = (2 ** 4)
    width = (width // base) * base + base * int(width > 0)
    height = (height // base) * base + base * int(height > 0)
    return height, width


def get_model_definition(img_height, img_width, in_channels, out_channels):
    base = (2 ** 4) + 1
    msg = "{actual} not multiple of " + str(base)
    # assert img_height % base == 0, msg.format(actual=img_height)
    # assert img_width % base == 0, msg.format(actual=img_width)
    inputs = Input((img_height, img_width, in_channels))
    x = Lambda(lambda aux: aux / 255)(inputs)
    kwargs_conv2d = {
        'kernel_size': (5, 5),
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'padding': 'same',
    }
    hdim = 100

    print(x.shape)
    x = Conv2D(hdim, **kwargs_conv2d)(x)
    x = Conv2D(hdim, **kwargs_conv2d)(x)
    x = Conv2D(hdim, **kwargs_conv2d)(x)
    x = tf.keras.layers.Cropping2D(cropping=(30, 70))(x)
    print(x.shape)
    x = Conv2D(hdim, **kwargs_conv2d)(x)
    x = Conv2D(hdim, **kwargs_conv2d)(x)
    x = tf.keras.layers.Cropping2D(cropping=(2, 28))(x)
    print(x.shape)
    x = Conv2D(hdim, **kwargs_conv2d)(x)

    outputs = Conv2D(out_channels, kernel_size=(1, 1), activation='sigmoid')(x)
    # Model compilation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    pre_process_input = identity_function
    return model, pre_process_input
