"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, Lambda, Input,
)


def identity_function(x):
    return x


def get_model_definition(img_height, img_width, in_channels, out_channels):
    base = 2
    msg = "{actual} not multiple of " + str(base)
    assert img_height % base == 0, msg.format(actual=img_height)
    assert img_width % base == 0, msg.format(actual=img_width)
    inputs = Input((img_height, img_width, in_channels))
    x = Lambda(lambda im: im / 255)(inputs)
    dim = 64
    k_size = (3, 3)
    x = Conv2D(dim, k_size, activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    x = Conv2D(dim, k_size, activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    x = Conv2D(dim, k_size, activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    outputs = Conv2D(out_channels, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    pre_process_input = identity_function
    return model, pre_process_input
