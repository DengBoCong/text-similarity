#! -*- coding: utf-8 -*-
""" Implementation of Char CNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def char_cnn(seq_len: int,
             embeddings_size: int,
             word_max_len: int,
             char_cnn_layers: list,
             highway_layers: int = 2,
             num_rnn_layers: int = 2,
             rnn_units: int = 650,
             dropout: float = 0.5,
             label_num: int = 2,
             label_act: str = "softmax") -> keras.Model:
    """Char CNN
    :param seq_len: 序列长度
    :param embeddings_size: 特征大小
    :param word_max_len: 单个token最大长度
    :param char_cnn_layers: 多层卷积列表，(filter_num, kernel_size_1)
            [[50, 1], [100, 2], [150, 3], [200, 4], [200, 5], [200, 6], [200, 7]]
    :param highway_layers: 使用highway层数
    :param num_rnn_layers: rnn层数
    :param rnn_units: rnn隐层大小
    :param dropout: 采样率
    :param label_num: 输出类别数
    :param label_act: 输出激活函数
    """
    embeddings = keras.Input(shape=(seq_len, embeddings_size))
    reshape_embeddings = tf.expand_dims(input=embeddings, axis=-1)
    concat_embeddings = keras.layers.Concatenate()([reshape_embeddings for i in range(word_max_len)])
    embeddings_outputs = tf.transpose(a=concat_embeddings, perm=[0, 1, 3, 2])
    conv_out = []
    for char_cnn_size in char_cnn_layers:
        conv = keras.layers.Conv2D(filters=char_cnn_size[0],
                                   kernel_size=(1, char_cnn_size[1]),
                                   activation="tanh")(embeddings_outputs)
        pooled = keras.layers.MaxPool2D(pool_size=(1, word_max_len - char_cnn_size[1] + 1))(conv)
        conv_out.append(pooled)

    outputs = keras.layers.Concatenate()(conv_out)
    outputs = keras.layers.Reshape(target_shape=(seq_len, outputs.shape[2] * sum(
        np.array([ccl[0] for ccl in char_cnn_layers])
    )))(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)

    for highway_layer in range(highway_layers):
        outputs = Highway()(outputs)

    for _ in range(num_rnn_layers):
        outputs = keras.layers.Bidirectional(keras.layers.LSTM(
            units=rnn_units, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.32 * 0.1),
            recurrent_regularizer=keras.regularizers.l2(0.32)
        ))(outputs)

        outputs = keras.layers.Dropout(rate=dropout)(outputs)

    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(units=label_num, activation=label_act)(outputs)

    return keras.Model(inputs=embeddings, outputs=outputs, name="char-cnn")


class Highway(keras.layers.Layer):
    """Highway
    """

    def __init__(self, transform_gate_bias: int = -2, **kwargs):
        """
        :param transform_gate_bias: 常数初始化器scalar
        """
        super(Highway, self).__init__(**kwargs)
        self.transform_gate_bias = transform_gate_bias

    def build(self, input_shape):
        super(Highway, self).build(input_shape)
        self.gate_transform = keras.layers.Dense(
            units=input_shape[-1], activation="sigmoid",
            bias_initializer=keras.initializers.Constant(self.transform_gate_bias)
        )
        self.block_state = keras.layers.Dense(
            units=input_shape[-1], activation="gelu", bias_initializer="zero"
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, *args, **kwargs):
        gate_transform = self.gate_transform(inputs)
        gate_cross = 1. - gate_transform
        block_state = self.block_state(inputs)
        highway = gate_transform * block_state + gate_cross * inputs
        return highway

    def get_config(self):
        config = super(Highway, self).get_config()
        config["transform_gate_bias"] = self.transform_gate_bias
        return config
