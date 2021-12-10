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


def rnn_layer(units: int,
              input_feature_dim: int,
              cell_type: str = "lstm",
              dropout: float = 0.0,
              if_bidirectional: bool = True,
              d_type: tf.dtypes.DType = tf.float32) -> tf.keras.Model:
    """ RNNCell层，其中可定义cell类型，是否双向
    :param units: cell单元数
    :param input_feature_dim: 输入的特征维大小
    :param cell_type: cell类型，lstm/gru， 默认lstm
    :param dropout: 采样率
    :param if_bidirectional: 是否双向
    :param d_type: 运算精度
    :return: Multi-layer RNN
    """
    inputs = tf.keras.Input(shape=(None, input_feature_dim), dtype=d_type)
    if cell_type == "lstm":
        rnn = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True,
                                   dropout=dropout, recurrent_initializer="glorot_uniform", dtype=d_type)
    elif cell_type == "gru":
        rnn = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True,
                                  dropout=dropout, recurrent_initializer="glorot_uniform", dtype=d_type)
    else:
        raise ValueError("{} is unknown type".format(cell_type))

    if if_bidirectional:
        rnn = tf.keras.layers.Bidirectional(layer=rnn, dtype=d_type)

    rnn_outputs = rnn(inputs)
    outputs = rnn_outputs[0]
    states = outputs[:, -1, :]

    return tf.keras.Model(inputs=inputs, outputs=[outputs, states])


def siamese_rnn_with_embedding(emb_dim: int,
                               vec_dim: int,
                               vocab_size: int,
                               units: int,
                               cell_type: str,
                               share: bool = True,
                               d_type: tf.dtypes.DType = tf.float32) -> tf.keras.Model:
    """ Siamese LSTM with Embedding
    :param emb_dim: embedding dim
    :param vec_dim: 特征维度大小
    :param vocab_size: 词表大小，例如为token最大整数index + 1.
    :param units: 输出空间的维度
    :param cell_type: RNN的实现类型
    :param share: 是否共享权重
    :param d_type: 运算精度
    :return: Model
    """
    input1 = tf.keras.Input(shape=(vec_dim,))
    input2 = tf.keras.Input(shape=(vec_dim,))

    embedding1 = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim,
                                           input_length=vec_dim, dtype=d_type)(input1)
    embedding2 = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim,
                                           input_length=vec_dim, dtype=d_type)(input2)

    if cell_type not in ["lstm", "gru"]:
        raise ValueError("{} is unknown type".format(cell_type))

    if share:
        if cell_type == "lstm":
            rnn_impl = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True, dtype=d_type)
        else:
            rnn_impl = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True, dtype=d_type)

        outputs1 = rnn_impl(embedding1)
        outputs2 = rnn_impl(embedding2)
    else:
        if cell_type == "lstm":
            rnn_impl1 = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True, dtype=d_type)
            rnn_impl2 = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True, dtype=d_type)
        else:
            rnn_impl1 = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True, dtype=d_type)
            rnn_impl2 = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True, dtype=d_type)

        outputs1 = rnn_impl1(embedding1)
        outputs2 = rnn_impl2(embedding2)

    return tf.keras.Model(inputs=[input1, input2], outputs=[outputs1[1], outputs2[1]])


def siamese_bi_rnn_with_embedding(emb_dim: int,
                                  vec_dim: int,
                                  vocab_size: int,
                                  dropout: float,
                                  num_layers: int,
                                  units: int,
                                  hidden_size: int,
                                  cell_type: str,
                                  d_type: tf.dtypes.DType = tf.float32) -> tf.keras.Model:
    """ Siamese LSTM with Embedding
    :param emb_dim: embedding dim
    :param vec_dim: 特征维度大小
    :param vocab_size: 词表大小，例如为token最大整数index + 1.
    :param dropout: 采样率
    :param num_layers: RNN层数
    :param units: 输出空间的维度
    :param hidden_size: rnn隐藏层维度
    :param cell_type: RNN的实现类型
    :param d_type: 运算精度
    :return: Model
    """
    input1 = tf.keras.Input(shape=(vec_dim,))
    input2 = tf.keras.Input(shape=(vec_dim,))

    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=vec_dim, dtype=d_type)

    output1 = embedding(input1)
    output2 = embedding(input2)

    if cell_type not in ["lstm", "gru"]:
        raise ValueError("{} is unknown type".format(cell_type))

    for i in range(num_layers):
        rnn = rnn_layer(units=hidden_size, input_feature_dim=emb_dim, cell_type=cell_type,
                        dropout=dropout, d_type=d_type, if_bidirectional=True)
        output1, state1 = rnn(output1)
        output2, state2 = rnn(output2)

    output1 = tf.math.reduce_sum(input_tensor=output1, axis=1)
    output2 = tf.math.reduce_sum(input_tensor=output2, axis=1)

    dense_layer = tf.keras.layers.Dense(units=units, activation="tanh")
    dropout_layer = tf.keras.layers.Dropout(rate=dropout)

    output1 = dense_layer(output1)
    output2 = dense_layer(output2)
    output1 = dropout_layer(output1)
    output2 = dropout_layer(output2)

    output = tf.keras.losses.cosine_similarity(y_true=output1, y_pred=output2, axis=-1)

    return tf.keras.Model(inputs=[input1, input2], outputs=output)
