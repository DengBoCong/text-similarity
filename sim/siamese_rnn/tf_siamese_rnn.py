#! -*- coding: utf-8 -*-
""" Implementation of Siamese RNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def siamese_rnn_with_embedding(emb_dim: int, vec_dim: int, vocab_size: int,
                               units: int, rnn: str, share: bool = True) -> tf.Model:
    """ Siamese LSTM with Embedding
    :param emb_dim: Dimension of the dense embedding
    :param vec_dim: Feature size, max length
    :param vocab_size: Size of the vocabulary, i.e. maximum integer index + 1.
    :param units: Dimensionality of the output space
    :param rnn: Types of RNN
    :param share: Whether to share weight
    :return: Model
    """
    input1 = tf.keras.Input(shape=(vec_dim,))
    input2 = tf.keras.Input(shape=(vec_dim,))

    embedding1 = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=vec_dim)
    embedding2 = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=vec_dim)

    if rnn not in ["lstm", "gru"]:
        raise ValueError("{} is unknown type".format(rnn))

    if share:
        if rnn == "lstm":
            rnn_impl = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
        else:
            rnn_impl = tf.keras.layers.GRU()

        outputs1 = rnn_impl(embedding1)
        outputs2 = rnn_impl(embedding2)
    else:
        if rnn == "lstm":
            rnn_impl1 = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
            rnn_impl2 = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
        else:
            rnn_impl1 = tf.keras.layers.GRU()
            rnn_impl2 = tf.keras.layers.GRU()

        outputs1 = rnn_impl1(embedding1)
        outputs2 = rnn_impl2(embedding2)

    return tf.keras.Model(inputs=[input1, input2], outputs=[outputs1[1], outputs2[1]])
