#! -*- coding: utf-8 -*-
""" Implementation of Fast Text
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras


def fast_text(units: int,
              embedding_size: int,
              label_size: int = 2,
              act: str = "tanh",
              dropout: float = 0.1) -> keras.Model:
    """ Fast Text Model
    :param units: 中间隐层hidden size
    :param embedding_size: 特征维度大小
    :param label_size: 类别数
    :param act: 激活函数
    :param dropout: 采样率
    """
    inputs = keras.Input(shape=(None, embedding_size))

    inputs_m = keras.layers.GlobalMaxPooling1D()(inputs)
    inputs_a = keras.layers.GlobalAveragePooling1D()(inputs)

    outputs = tf.concat([inputs_m, inputs_a], axis=-1)
    outputs = keras.layers.Dense(units=units, activation=act)(outputs)
    outputs = keras.layers.Dropout(rate=dropout)(outputs)
    outputs = keras.layers.Dense(units=label_size, activation="softmax")(outputs)

    return keras.Model(inputs=inputs, outputs=outputs, name="fast-text")
