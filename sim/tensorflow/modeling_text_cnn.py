#! -*- coding: utf-8 -*-
""" Implementation of Text CNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras


def text_cnn(seq_len: int,
             embeddings_size: int,
             units: int,
             filter_num: int,
             kernel_sizes: list,
             initializers: list,
             activations: list,
             padding: str = "valid",
             dropout: float = 0.1,
             act: str = "tanh") -> keras.Model:
    """Text CNN
    :param seq_len: 序列长度
    :param embeddings_size: 特征大小
    :param units: 全连接层hidden size
    :param filter_num: 滤波器个数
    :param kernel_sizes: 卷积核大小，可以是多个
    :param initializers: 卷积核初始化器列表，个数同kernel_sizes
    :param activations: 激活函数列表，个数同kernel_sizes
    :param padding: 填充类型
    :param dropout: 采样率
    :param act: 全连接层激活函数
    """
    embeddings = keras.Input(shape=(seq_len, embeddings_size))
    conv_pools = []
    reshape_embeddings = tf.expand_dims(input=embeddings, axis=-1)
    for kernel_size, initializer, activation in zip(kernel_sizes, initializers, activations):
        conv = keras.layers.Conv2D(
            filters=filter_num, kernel_size=(kernel_size, embeddings_size), strides=(1, 1),
            padding=padding, kernel_initializer=initializer, activation=activation
        )(reshape_embeddings)

        pooled = keras.layers.MaxPool2D(pool_size=(seq_len - kernel_size + 1, 1),
                                        strides=(1, 1), padding="valid")(conv)

        conv_pools.append(pooled)

    outputs = keras.layers.Concatenate(axis=-1)(conv_pools)
    outputs = keras.layers.Dropout(rate=dropout)(outputs)
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(units=units, activation=act)(outputs)

    return keras.Model(inputs=embeddings, outputs=outputs, name="text-cnn")
