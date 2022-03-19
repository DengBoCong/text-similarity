#! -*- coding: utf-8 -*-
""" Implementation of Text VDCNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras
from typing import Any


class KMaxPooling(keras.layers.Layer):
    """动态K-max pooling
     k的选择为 k = max(k, s * (L-1) / L)
     其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
    """

    def __init__(self, top_k: int = 8, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.top_k = top_k

    def build(self, input_shape):
        super(KMaxPooling, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        reshape_inputs = tf.transpose(a=inputs, perm=[0, 2, 1])
        pool_top_k = tf.nn.top_k(input=reshape_inputs, k=self.top_k, sorted=False).values
        reshape_pool_top_k = tf.transpose(a=pool_top_k, perm=[0, 2, 1])
        return reshape_pool_top_k

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.top_k, input_shape[-1]


def convolutional_block(inputs: Any,
                        filter_num: int = 256,
                        kernel_size: int = 3,
                        strides: Any = 1,
                        padding: str = "same",
                        l2: float = 0.0000032,
                        activation: Any = "linear") -> Any:
    """ Convolutional block
    :param inputs: 输入
    :param filter_num: 滤波器大小
    :param kernel_size: 卷积核大小
    :param strides: 移动步幅
    :param padding: 填充类型
    :param l2: 正则化因子
    :param activation: 激活函数
    """
    outputs = keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size, padding=padding,
                                  strides=strides, kernel_regularizer=keras.regularizers.l2(l2),
                                  bias_regularizer=keras.regularizers.l2(l2), activation=activation)(inputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.ReLU()(outputs)
    outputs = keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size, strides=strides,
                                  padding=padding, kernel_regularizer=keras.regularizers.l2(l2),
                                  bias_regularizer=keras.regularizers.l2(l2), activation=activation)(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.ReLU()(outputs)
    return outputs


def down_sampling(inputs: Any,
                  pool_type: str = "max",
                  pool_size: Any = 3,
                  strides: Any = 2,
                  padding: str = "same", ) -> Any:
    """
    :param inputs: 输入
    :param pool_type: "max", "k-max", "conv"
    :param pool_size: 池化窗口大小
    :param strides: 移动步幅
    :param padding: 填充类型
    """
    if pool_type == "max":
        outputs = keras.layers.MaxPool1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    elif pool_type == "k-max":
        outputs = KMaxPooling(top_k=inputs.shape[1] // 2)(inputs)
    elif pool_type == "conv":
        outputs = keras.layers.Conv1D(kernel_size=pool_size, strides=strides, padding=padding)(inputs)
    else:
        outputs = keras.layers.MaxPool1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)

    return outputs


def shortcut_pool(inputs: Any,
                  inputs_mid: Any,
                  filter_num: int = 256,
                  kernel_size: Any = 1,
                  strides: Any = 2,
                  padding: str = "same",
                  pool_type: str = "max",
                  shortcut: bool = True) -> Any:
    """shortcut连接. 恒等映射, block+f(block)，加上 down sampling
    :param inputs: block
    :param inputs_mid: conv block outputs
    :param filter_num: 滤波器大小
    :param kernel_size: 卷积核大小
    :param strides: 移动步幅
    :param padding: 填充类型
    :param pool_type: "max", "k-max", "conv"
    :param shortcut: 是否开启shortcut连接
    """
    if shortcut:
        conv = keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size,
                                   strides=strides, padding=padding)(inputs)
        conv = keras.layers.BatchNormalization()(conv)
        outputs = down_sampling(inputs_mid, pool_type=pool_type)
        outputs = conv + outputs
    else:
        outputs = keras.layers.ReLU(inputs)
        outputs = down_sampling(outputs, pool_type=pool_type)

    if pool_type is not None:
        outputs = keras.layers.Conv1D(filters=filter_num * 2, kernel_size=kernel_size,
                                      strides=1, padding=padding)(outputs)
        outputs = keras.layers.BatchNormalization()(outputs)

    return outputs


def text_vdcnn(seq_len: int,
               embeddings_size: int,
               filters: list,
               dropout_spatial: float = 0.2,
               dropout: float = 0.32,
               l2: float = 0.0000032,
               activation_conv: Any = "linear",
               pool_type: str = "max",
               top_k: int = 2,
               label_num: int = 2,
               activate_classify: str = "softmax") -> keras.Model:
    """Text VDCNN
    :param seq_len: 序列长度
    :param embeddings_size: 特征大小
    :param filters: 滤波器配置, eg: [[64, 1], [128, 1], [256, 1], [512, 1]]
    :param dropout_spatial: 空间采样率
    :param dropout: 采样率
    :param l2: 正则化因子
    :param activation_conv: 激活函数
    :param pool_type: "max", "k-max", "conv"
    :param top_k: max(k, s * (L-1) / L)
    :param label_num: 类别数
    :param activate_classify: 分类层激活函数
    """
    inputs = keras.Input(shape=(seq_len, embeddings_size))
    embeddings = keras.layers.SpatialDropout1D(rate=dropout_spatial)(inputs)

    conv = keras.layers.Conv1D(
        filters=filters[0][0], kernel_size=1, strides=1, padding="same", kernel_regularizer=keras.regularizers.l2(l2),
        bias_regularizer=keras.regularizers.l2(l2), activation=activation_conv
    )(embeddings)
    block = keras.layers.ReLU()(conv)

    for filters_block in filters:
        for j in range(filters_block[1] - 1):
            block_mid = convolutional_block(block, filter_num=filters_block[0])
            block = block + block_mid

        # 这里是conv + max-pooling
        block_mid = convolutional_block(block, filter_num=filters_block[0])
        block = shortcut_pool(block, block_mid, filter_num=filters_block[0], pool_type=pool_type)

    block = KMaxPooling(top_k=top_k)(block)
    block = keras.layers.Flatten()(block)
    block = keras.layers.Dropout(rate=dropout)(block)

    outputs = keras.layers.Dense(units=label_num, activation=activate_classify)(block)
    return keras.Model(inputs=inputs, outputs=outputs, name="vdcnn")
