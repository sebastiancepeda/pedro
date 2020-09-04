"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import tensorflow as tf


class CNN_Encoder(tf.keras.Model):

    def __init__(self, embedding_dim, img_height, img_width):
        super(CNN_Encoder, self).__init__()
        base = (2 ** 4)
        msg = "{actual} not multiple of " + str(base)
        assert img_height % base == 0, msg.format(actual=img_height)
        assert img_width % base == 0, msg.format(actual=img_width)
        self.l_preprocessing = tf.keras.layers.Lambda(lambda aux: aux / 255)
        kwargs_conv2d = {
            'filters': 10,
            'kernel_size': (3,) * 2,
            'activation': tf.keras.layers.LeakyReLU(alpha=0.1),
            'kernel_initializer': 'he_normal',
            'padding': 'same',
        }
        self.l_1 = tf.keras.layers.Conv2D(**kwargs_conv2d)
        self.l_2 = tf.keras.layers.Conv2D(**kwargs_conv2d)
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc_1 = tf.keras.layers.Dense(embedding_dim)
        self.fc_2 = tf.nn.relu

    def call(self, x):
        x = self.l_preprocessing(x)
        x = self.l_1(x)
        x = self.l_2(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x
