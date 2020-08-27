"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

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
    k_size = (7, 7)
    x = Conv2D(200, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(100, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(50, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(100, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(25, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(50, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(25, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(50, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(25, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(50, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(25, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(50, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(25, kernel_size=k_size, **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(50, kernel_size=k_size, **kwargs_conv2d)(x)
    x = Conv2D(25, kernel_size=k_size, **kwargs_conv2d)(x)
    outputs = Conv2D(out_channels, kernel_size=(1, 1), activation='sigmoid')(x)
    # Model compilation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    pre_process_input = identity_function
    return model, pre_process_input
