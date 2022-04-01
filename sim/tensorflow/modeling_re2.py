#! -*- coding: utf-8 -*-
""" Tensorflow Simple-Effective-Text-Matching Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.keras as keras
from typing import Any


def residual(inputs: Any, res_inputs: Any, _) -> Any:
    """残差"""
    if inputs.shape[-1] != res_inputs.shape[-1]:
        inputs = keras.layers.Dense(units=res_inputs.shape[-1])(inputs)
    return (inputs + res_inputs) * tf.sqrt(x=0.5)


def augmented_residual(inputs: Any, res_inputs: Any, index: int) -> Any:
    """增强残差"""
    outputs = inputs
    if index == 1:
        outputs = tf.concat(values=[res_inputs, inputs], axis=-1)
    elif index > 1:
        hidden_size = inputs.shape[-1]
        outputs = (res_inputs[:, :, -hidden_size:] + inputs) * tf.sqrt(x=0.5)
        outputs = tf.concat(values=[res_inputs[:, :, :-hidden_size], outputs], axis=-1)

    return outputs


def re2_encoder(embedding_size: int,
                filters_num: int,
                enc_layers: int = 2,
                kernel_size: Any = 3,
                dropout: float = 0.8,
                name: str = "re2-encoder") -> keras.Model:
    """RE2 Encoder
    :param embedding_size: feature size
    :param filters_num: filter size
    :param enc_layers: encoder layer num
    :param kernel_size: 卷积核大小
    :param dropout: 采样率
    :param name: 模型名
    """
    inputs = keras.Input(shape=(None, embedding_size))
    mask = keras.Input(shape=(None, 1))

    outputs = inputs
    for enc_index in range(enc_layers):
        outputs = mask * outputs
        if enc_index > 0:
            outputs = keras.layers.Dropout(rate=dropout)(outputs)
        outputs = keras.layers.Conv1D(filters=filters_num, kernel_size=kernel_size,
                                      padding="same", activation="relu")(outputs)

    outputs = keras.layers.Dropout(rate=dropout)(outputs)

    return keras.Model(inputs=[inputs, mask], outputs=outputs, name=name)


class Alignment(keras.layers.Layer):
    """对齐层"""

    def __init__(self, hidden_size: int, dropout: float, align_type: str = "linear", **kwargs):
        """
        :param hidden_size: feature size
        :param dropout: 采样率
        :param align_type: 对齐方式，identity/linear
        """
        super(Alignment, self).__init__(**kwargs)
        if align_type == "linear":
            self.linear_dropout1 = keras.layers.Dropout(rate=dropout)
            self.linear_dense1 = keras.layers.Dense(units=hidden_size, activation="relu")
            self.linear_dropout2 = keras.layers.Dropout(rate=dropout)
            self.linear_dense2 = keras.layers.Dense(units=hidden_size, activation="relu")

        self.hidden_size = hidden_size
        self.align_type = align_type

    def build(self, input_shape):
        super(Alignment, self).build(input_shape)
        self.temperature = self.add_weight(name="temperature", shape=(), dtype=tf.float32,
                                           initializer=tf.constant_initializer(value=math.sqrt(1 / self.hidden_size)))

    def call(self, inputs, *args, **kwargs):
        a_inputs, a_mask, b_inputs, b_mask = inputs

        # Attention
        if self.align_type == "identity":
            attention_outputs = tf.matmul(a=a_inputs, b=b_inputs, transpose_b=True) * self.temperature
        elif self.align_type == "linear":
            a_outputs = self.linear_dropout1(a_inputs)
            a_outputs = self.linear_dense1(a_outputs)
            b_outputs = self.linear_dropout2(b_inputs)
            b_outputs = self.linear_dense2(b_outputs)
            attention_outputs = tf.matmul(a=a_outputs, b=b_outputs, transpose_b=True) * self.temperature
        else:
            raise ValueError("`align_type` must be identity or linear")

        attention_mask = tf.matmul(a=a_mask, b=b_mask, transpose_b=True)
        attention_outputs = attention_mask * attention_outputs + (1 - attention_mask) * tf.float32.min
        a_attention = keras.layers.Softmax(axis=1)(attention_outputs)
        b_attention = keras.layers.Softmax(axis=2)(attention_outputs)

        a_feature = tf.matmul(a=a_attention, b=a_inputs, transpose_a=True)
        b_feature = tf.matmul(a=b_attention, b=b_inputs)

        return a_feature, b_feature


class Fusion(keras.layers.Layer):
    """Fusion Layer
    """

    def __init__(self, hidden_size: int, dropout: float, fusion_type: str = "full", **kwargs):
        """
        :param hidden_size: feature size
        :param dropout: 采样率
        :param fusion_type: fusion type，simple/full
        """
        super(Fusion, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fusion_type = fusion_type

    def build(self, input_shape):
        super(Fusion, self).build(input_shape)
        if self.fusion_type == "full":
            self.orig_dense = keras.layers.Dense(units=self.hidden_size, activation="relu")
            self.sub_dense = keras.layers.Dense(units=self.hidden_size, activation="relu")
            self.mul_dense = keras.layers.Dense(units=self.hidden_size, activation="relu")
            self.dropout_layer = keras.layers.Dropout(rate=self.dropout)
            self.output_dense = keras.layers.Dense(units=self.hidden_size, activation="relu")
        elif self.fusion_type == "simple":
            self.dense = keras.layers.Dense(units=self.hidden_size, activation="relu")
        else:
            raise ValueError("`fusion_type` must be full or simple")

    def call(self, inputs, *args, **kwargs):
        inputs, align_inputs = inputs
        if self.fusion_type == "full":
            outputs = tf.concat(values=[
                self.orig_dense(tf.concat(values=[inputs, align_inputs], axis=-1)),
                self.sub_dense(tf.concat(values=[inputs, inputs - align_inputs], axis=-1)),
                self.mul_dense(tf.concat(values=[inputs, inputs * align_inputs], axis=-1))
            ], axis=-1)
            outputs = self.dropout_layer(outputs)
            outputs = self.output_dense(outputs)
        else:
            outputs = self.dense(tf.concat(values=[inputs, align_inputs], axis=-1))

        return outputs


class Prediction(keras.layers.Layer):
    """Prediction Layer
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout: float, pred_type: str = "full", **kwargs):
        """
        :param num_classes: 类别数
        :param hidden_size: feature size
        :param dropout: 采样率
        :param pred_type: prediction type，simple/full/symmetric
        """
        super(Prediction, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pred_type = pred_type

    def build(self, input_shape):
        self.dropout1 = keras.layers.Dropout(rate=self.dropout)
        self.dense1 = keras.layers.Dense(units=self.hidden_size, activation="relu")
        self.dropout2 = keras.layers.Dropout(rate=self.dropout)
        self.dense2 = keras.layers.Dense(units=self.num_classes)

    def call(self, inputs, *args, **kwargs):
        a_feature, b_feature = inputs
        if self.pred_type == "simple":
            outputs = tf.concat(values=[a_feature, b_feature], axis=-1)
        elif self.pred_type == "full":
            outputs = tf.concat(values=[a_feature, b_feature, a_feature * b_feature, a_feature - b_feature], axis=-1)
        elif self.pred_type == "symmetric":
            outputs = tf.concat(values=[a_feature, b_feature, a_feature * b_feature,
                                        tf.abs(x=(a_feature - b_feature))], axis=-1)
        else:
            raise ValueError("`pred_type` must be simple, full or symmetric")

        outputs = self.dropout1(outputs)
        outputs = self.dense1(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.dense2(outputs)

        return outputs


def re2_network(vocab_size: int,
                embedding_size: int,
                block_layer_num: int = 2,
                enc_layers: int = 2,
                enc_kernel_size: Any = 3,
                dropout: float = 0.8,
                num_classes: int = 2,
                hidden_size: int = None,
                connection_args: str = "aug",
                align_type: str = "linear",
                fusion_type: str = "full",
                pred_type: str = "full") -> keras.Model:
    """Simple-Effective-Text-Matching
    :param vocab_size: 词表大小
    :param embedding_size: feature size
    :param block_layer_num: fusion block num
    :param enc_layers: encoder layer num
    :param enc_kernel_size: 卷积核大小
    :param dropout: 采样率
    :param num_classes: 类别数
    :param hidden_size: 隐藏层大小
    :param connection_args: 连接层模式，residual/aug
    :param align_type: 对齐方式，identity/linear
    :param fusion_type: fusion type，simple/full
    :param pred_type: prediction type，simple/full/symmetric
    """
    text_a_input_ids = keras.Input(shape=(None,))
    text_a_mask = tf.cast(x=tf.math.equal(text_a_input_ids, 0), dtype=tf.float32)[:, :, tf.newaxis]

    text_b_input_ids = keras.Input(shape=(None,))
    text_b_mask = tf.cast(x=tf.math.equal(text_b_input_ids, 0), dtype=tf.float32)[:, :, tf.newaxis]

    if not hidden_size:
        hidden_size = embedding_size // 2
    embeddings = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
    embeddings_dropout = keras.layers.Dropout(rate=dropout)
    connection = {
        "residual": residual,
        "aug": augmented_residual
    }

    a_embeddings = embeddings(text_a_input_ids)
    a_outputs = embeddings_dropout(a_embeddings)
    b_embeddings = embeddings(text_b_input_ids)
    b_outputs = embeddings_dropout(b_embeddings)

    a_residual, b_residual = a_outputs, b_outputs

    for index in range(block_layer_num):
        if index > 0:
            a_outputs = connection[connection_args](a_outputs, a_residual, index)
            b_outputs = connection[connection_args](b_outputs, b_residual, index)
            a_residual, b_residual = a_outputs, b_outputs

        # Encoder
        encoder = re2_encoder(a_outputs.shape[-1], hidden_size, enc_layers,
                              enc_kernel_size, dropout, f"re2-encoder-{index}")
        a_encoder_outputs = encoder([a_outputs, text_a_mask])
        b_encoder_outputs = encoder([b_outputs, text_b_mask])

        # cat
        a_outputs = tf.concat(values=[a_outputs, a_encoder_outputs], axis=-1)
        b_outputs = tf.concat(values=[b_outputs, b_encoder_outputs], axis=-1)

        # alignment
        a_align, b_align = Alignment(hidden_size=hidden_size, dropout=dropout,
                                     align_type=align_type)([a_outputs, text_a_mask, b_outputs, text_b_mask])
        fusion_layer = Fusion(hidden_size=hidden_size, dropout=dropout, fusion_type=fusion_type)
        a_outputs = fusion_layer([a_outputs, a_align])
        b_outputs = fusion_layer([b_outputs, b_align])

    a_outputs = tf.reduce_max(input_tensor=text_a_mask * a_outputs + (1. - text_a_mask) * -1e9, axis=1)
    b_outputs = tf.reduce_max(input_tensor=text_b_mask * b_outputs + (1. - text_b_mask) * -1e9, axis=1)

    outputs = Prediction(num_classes=num_classes, hidden_size=hidden_size,
                         dropout=dropout, pred_type=pred_type)([a_outputs, b_outputs])

    return keras.Model(inputs=[text_a_input_ids, text_b_input_ids], outputs=outputs, name="re2")
