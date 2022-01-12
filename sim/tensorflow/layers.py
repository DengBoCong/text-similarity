#! -*- coding: utf-8 -*-
""" Tensorflow Common Modules
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


class PositionEmbedding(keras.layers.Layer):
    """定义可训练的位置Embedding
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 merge_mode: str = "add",
                 hierarchical: Any = None,
                 custom_position_ids: bool = False,
                 embeddings_initializer: Any = "zeros",
                 **kwargs):
        """
        :param input_dim: 输入维度
        :param output_dim: 输出维度
        :param merge_mode: 输入和position合并的方式
        :param hierarchical: 是否层次分解位置编码
        :param custom_position_ids: 是否传入自定义位置编码id
        :param embeddings_initializer: 初始化器
        """
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.custom_position_ids = custom_position_ids
        self.embeddings_initializer = embeddings_initializer
        self.embeddings = None

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs, *args, **kwargs):
        """如果传入自定义position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if "int" not in position_ids.dtype.name:
                position_ids = tf.cast(x=position_ids, dtype=tf.int32)
        else:
            batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
            position_ids = tf.expand_dims(input=tf.range(start=0, limit=seq_len, delta=1), axis=0)

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = tf.gather(params=embeddings, indices=position_ids // self.input_dim)
            embeddings_y = tf.gather(params=embeddings, indices=position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = tf.gather(params=self.embeddings, indices=position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == "add":
            return inputs + embeddings
        elif self.merge_mode == "mul":
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == "zero":
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = tf.tile(input=embeddings, multiples=[batch_size, 1, 1])
            return tf.concat(values=[inputs, embeddings], axis=-1)

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "merge_model": self.merge_mode,
            "hierarchical": self.hierarchical,
            "embeddings_initializer": keras.initializers.serialize(initializer=self.embeddings_initializer),
            "custom_position_ids": self.custom_position_ids
        }
        base_config = super(PositionEmbedding, self).get_config()
        base_config.update(config)
        return base_config


def scaled_dot_product_attention(query: tf.Tensor,
                                 key: tf.Tensor,
                                 value: tf.Tensor,
                                 hidden_size: int,
                                 attention_head_size: int,
                                 dropout: float,
                                 is_training: bool,
                                 mask: Any = None,
                                 manual_seed: int = 1) -> tuple:
    """点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param hidden_size: hidden size
    :param attention_head_size: 分头之后维度大小
    :param dropout: 注意力dropout
    :param is_training: 是否处于训练模式
    :param mask: float, (..., seq_len_q, seq_len_k)
    :param manual_seed: 随机种子
    """
    batch_size = tf.shape(query)[0]
    attention_scores = tf.matmul(a=query, b=key, transpose_b=True)
    attention_scores = attention_scores / tf.math.sqrt(x=attention_head_size)

    if mask is not None:
        attention_scores += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits=attention_scores, axis=-1)
    attention_weights = keras.layers.Dropout(rate=dropout, seed=manual_seed)(attention_weights, is_training)

    context_layer = tf.matmul(a=attention_weights, b=value)
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
    context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, hidden_size))

    return context_layer, attention_weights


def transpose_for_scores(input_tensor: tf.Tensor, head_num: int, head_size: int):
    """分拆最后一个维度到 (num_heads, depth)
    :param input_tensor: 输入
    :param head_num: 注意力头数
    :param head_size: 每个注意力头维数
    """
    batch_size = input_tensor.shape[0]
    input_tensor = tf.reshape(input_tensor, (batch_size, -1, head_num, head_size))
    return tf.transpose(input_tensor, perm=[0, 2, 1, 3])


def bert_self_attention(num_heads: int,
                        head_size: int,
                        attention_func: Any,
                        is_training: bool,
                        attention_dropout: float,
                        use_bias: bool = True,
                        key_size: int = None,
                        hidden_size: int = None,
                        initializer: Any = "glorot_uniform",
                        manual_seed: int = 1,
                        name: str = "multi-head-self-attention") -> keras.Model:
    """Bert Self-Attention
    :param num_heads: 注意力头数
    :param head_size: Attention中V的head_size
    :param attention_func: 注意力计算方法
    :param is_training: 是否处于训练模式
    :param attention_dropout: Attention矩阵的Dropout比例
    :param use_bias: 是否加上偏差项
    :param key_size: Attention中Q,K的head_size
    :param hidden_size: 编码维度
    :param initializer: 初始化器
    :param manual_seed: 随机种子
    :param name: 模型名
    """
    query_inputs = keras.Input(shape=(None, None))
    key_inputs = keras.Input(shape=(None, None))
    value_inputs = keras.Input(shape=(None, None))
    mask = keras.Input(shape=(None, None, None))
    key_size = key_size if key_size is not None else head_size
    hidden_size = hidden_size if hidden_size is not None else num_heads * head_size

    query = keras.layers.Dense(units=key_size * num_heads, use_bias=use_bias,
                               kernel_initializer=initializer)(query_inputs)
    key = keras.layers.Dense(units=key_size * num_heads, use_bias=use_bias,
                             kernel_initializer=initializer)(key_inputs)
    value = keras.layers.Dense(units=head_size * num_heads, use_bias=use_bias,
                               kernel_initializer=initializer)(value_inputs)

    query = transpose_for_scores(input_tensor=query, head_num=num_heads, head_size=key_size)
    key = transpose_for_scores(input_tensor=key, head_num=num_heads, head_size=key_size)
    value = transpose_for_scores(input_tensor=value, head_num=num_heads, head_size=head_size)

    scaled_attention, attention_weights = attention_func(
        query=query,
        key=key,
        value=value,
        hidden_size=hidden_size,
        attention_head_size=head_size,
        dropout=attention_dropout,
        is_training=is_training,
        mask=mask,
        manual_seed=manual_seed
    )

    attn_outputs = keras.layers.Dense(units=hidden_size, use_bias=use_bias,
                                      kernel_initializer=initializer)(scaled_attention)

    return keras.Model(inputs=[query_inputs, key_inputs, value_inputs, mask],
                       outputs=[attn_outputs, attention_weights], name=name)


def feed_forward(units: int,
                 activation: Any = "gelu",
                 use_bias: bool = True,
                 kernel_initializer: Any = "glorot_uniform",
                 name: str = "feedforward") -> keras.Model:
    """FeedForward层
    https://arxiv.org/abs/2002.05202
    :param units: 输出维度
    :param use_bias: 是否使用偏差项
    :param activation: 激活函数，如果传入的是list，则将使用门控线性单元
    :param kernel_initializer: 初始化器
    :param name: 模型名
    """
    inputs = keras.Input(shape=(None, None))

    if not isinstance(activation, list):
        activation = [activation]

    outputs = keras.layers.Dense(
        units=units,
        activation=activation[0],
        use_bias=use_bias,
        kernel_initializer=kernel_initializer
    )(inputs)

    for index in range(1, len(activation)):
        outputs = keras.layers.Dense(
            units=units,
            activation=activation[index],
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )(outputs)

    outputs = keras.layers.Dense(
        units=units,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer
    )(outputs)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
