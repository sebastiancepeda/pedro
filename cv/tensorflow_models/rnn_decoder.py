"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import tensorflow as tf

from cv.tensorflow_models.bahdanau_attention import BahdanauAttention


class RNN_Decoder(tf.keras.Model):

    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # x shape after embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concat == (batch_size, 1, embedding_dim + hidden_size)
        # print("x1", x.shape)
        # print("context_vector", context_vector.shape)
        context_vector = tf.reshape(context_vector, [1, context_vector.shape[1]*context_vector.shape[2]])
        # print("context_vector", context_vector.shape)
        context_vector = tf.expand_dims(context_vector, 1)
        # print("context_vector", context_vector.shape)
        x = tf.concat([context_vector, x], axis=-1)
        # print("x2", x.shape)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        # print("x3", x.shape)
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        # print("x4", x.shape)
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        # print("x5", x.shape)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        zeroes = tf.zeros((batch_size, self.units))
        return zeroes
