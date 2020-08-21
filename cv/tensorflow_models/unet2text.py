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
    pre_processing = Lambda(lambda x: x / 255)(inputs)
    kwargs_conv2d = {
        'kernel_size': (3, 3),
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'padding': 'same',
    }
    h_dim = 20  # 100
    outs = {
        1: h_dim,  # 64
        2: h_dim,
        3: h_dim,
        4: h_dim,
        5: h_dim,
    }
    # Down
    c1 = Conv2D(outs[1], **kwargs_conv2d)(pre_processing)
    c1 = Conv2D(outs[1], **kwargs_conv2d)(c1)
    c2 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(outs[2], **kwargs_conv2d)(c2)
    c2 = Conv2D(outs[2], **kwargs_conv2d)(c2)
    c3 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(outs[3], **kwargs_conv2d)(c3)
    c3 = Conv2D(outs[3], **kwargs_conv2d)(c3)
    c4 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(outs[4], **kwargs_conv2d)(c4)
    c4 = Conv2D(outs[4], **kwargs_conv2d)(c4)
    c5 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(outs[5], **kwargs_conv2d)(c5)
    c5 = Conv2D(outs[5], **kwargs_conv2d)(c5)
    # Up
    u4 = Conv2DTranspose(outs[4], (2, 2), strides=(2, 2), padding='same')(c5)
    u4 = concatenate([u4, c4])
    u4 = Conv2D(outs[4], **kwargs_conv2d)(u4)
    u4 = Conv2D(outs[4], **kwargs_conv2d)(u4)
    u3 = Conv2DTranspose(outs[3], (2, 2), strides=(2, 2), padding='same')(u4)
    u3 = concatenate([u3, c3])
    u3 = Conv2D(outs[3], **kwargs_conv2d)(u3)
    u3 = Conv2D(outs[3], **kwargs_conv2d)(u3)
    u2 = Conv2DTranspose(outs[2], (2, 2), strides=(2, 2), padding='same')(u3)
    u2 = concatenate([u2, c2])
    u2 = Conv2D(outs[2], **kwargs_conv2d)(u2)
    u2 = Conv2D(outs[2], **kwargs_conv2d)(u2)
    u1 = Conv2DTranspose(outs[1], (2, 2), strides=(2, 2), padding='same')(u2)
    u1 = concatenate([u1, c1], axis=3)
    u1 = Conv2D(outs[1], **kwargs_conv2d)(u1)
    u1 = Conv2D(outs[1], **kwargs_conv2d)(u1)
    # Downward
    kwargs_conv2d = {
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'padding': 'same',
    }
    # h_dim = 100
    x = Conv2D(h_dim, kernel_size=(3, 3), **kwargs_conv2d)(u1)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=(3, 3), **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=(3, 3), **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=(3, 3), **kwargs_conv2d)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(h_dim, kernel_size=(3, 3), **kwargs_conv2d)(x)
    x = MaxPooling2D((4, 1))(x)
    x = Conv2D(h_dim, kernel_size=(1, 3), **kwargs_conv2d)(x)
    outputs = Conv2D(out_channels, (1, 1), activation='sigmoid')(x)
    # Model compilation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    pre_process_input = identity_function
    return model, pre_process_input
