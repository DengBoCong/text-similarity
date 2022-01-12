#! -*- coding: utf-8 -*-
""" Implementation of Bert
"""
# Author: DengBoCong <bocongdeng@gmail.com>
# https://arxiv.org/pdf/1706.03762.pdf
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
from sim.bert_base import BertConfig
from typing import Any


def embedding_lookup(input_ids: tf.Tensor,
                     vocab_size: int,
                     embedding_size: int = 128,
                     initializer_range: float = 0.02,
                     use_one_hot_embeddings: bool = False) -> tuple:
    """自训练embedding_lookup
    :param input_ids: int32, [batch_size, seq_length]
    :param vocab_size: 词汇表大小
    :param embedding_size: embedding维度
    :param initializer_range: embedding初始值范围
    :param use_one_hot_embeddings: 是否使用one-hot的方式查找Embedding，猜测one-hot的方式更加适合TPU计算
    """
    input_ids = tf.expand_dims(input_ids, axis=-1)
    embedding_table = tf.Variable(
        tf.random.truncated_normal(shape=[vocab_size, embedding_size], stddev=initializer_range)
    )
    flat_input_ids = tf.reshape(tensor=input_ids, shape=[-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(indices=flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = output.shape.as_list()
    output = tf.reshape(tensor=output, shape=input_shape[0: -1] + [input_shape[-1] * embedding_size])
    return output, embedding_table


def embedding_postprocessor(input_tensor: tf.Tensor,
                            use_token_type: bool = False,
                            token_type_ids: tf.Tensor = None,
                            token_type_vocab_size: int = 2,
                            use_position_embeddings: bool = True,
                            initializer_range: float = 0.02,
                            max_position_embeddings: int = 512,
                            dropout_prob: float = 0.1,
                            manual_seed: int = 1,
                            training: bool = True) -> tf.Tensor:
    """对embedding进行进一步处理
    :param input_tensor: float, [batch_size, seq_length, embedding_size]
    :param use_token_type: 是否加入token type embedding
    :param token_type_ids: token type embedding
    :param token_type_vocab_size: token type数量
    :param use_position_embeddings: 是否使用位置embedding
    :param initializer_range: embedding初始值范围
    :param max_position_embeddings: position embedding 维度
    :param dropout_prob: 应用在最后的输出的dropout
    :param manual_seed: 随机种子
    :param training: 是否是训练模式
    """
    batch_size, seq_len, width = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]
    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if `use_token_type` is True")

        token_type_table = tf.Variable(
            tf.random.truncated_normal(shape=[token_type_vocab_size, width], stddev=initializer_range)
        )
        # 这里使用one-hot方式查表，因为vocab较小，使用one-hot速度更快
        flat_token_type_ids = tf.reshape(tensor=token_type_ids, shape=[-1])
        one_hot_ids = tf.one_hot(indices=flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(tensor=token_type_embeddings, shape=[batch_size, seq_len, width])

        output += token_type_embeddings

    if use_position_embeddings:
        full_position_embeddings = tf.Variable(
            tf.random.truncated_normal(shape=[max_position_embeddings, width], stddev=initializer_range)
        )
        position_embeddings = tf.slice(input_=full_position_embeddings, begin=[0, 0], size=[seq_len, -1])

        num_dims = len(output.shape.as_list())
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([seq_len, width])
        position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)

        output += position_embeddings

    output = tf.keras.layers.LayerNormalization(axis=1)(output)
    output = tf.keras.layers.Dropout(rate=dropout_prob, seed=manual_seed)(output, training)

    return output


def _get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> Tuple:
    """pos/10000^(2i/d_model)
    :param pos: 字符总的数量按顺序递增
    :param i: 词嵌入大小按顺序递增
    :param d_model: 词嵌入大小
    :return: shape=(pos.shape[0], d_model)
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def gen_relative_positions_embeddings(position: int, d_model: int) -> tuple:
    """PE(pos,2i) = sin(pos/10000^(2i/d_model)) | PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    :param position: 字符总数
    :param d_model: 词嵌入大小
    """
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=d_type)














def bert_self_output(config: BertConfig, is_training: bool, manual_seed: int = 1) -> tf.keras.Model:
    """Bert Self-Attention Output
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param manual_seed: 随机种子
    """
    hidden_states = tf.keras.Input(shape=(None, None))
    input_tensor = tf.keras.Input(shape=(None, None))

    hidden_states = tf.keras.layers.Dense(units=config.hidden_size)(hidden_states)
    hidden_states = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob,
                                            seed=manual_seed)(hidden_states, is_training)
    hidden_states = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(hidden_states + input_tensor)

    return tf.keras.Model(inputs=[hidden_states, input_tensor], outputs=hidden_states)








def bert_encoder(config: BertConfig, is_training: bool, manual_seed: int = 1) -> tf.keras.Model:
    """Bert Encoder
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param manual_seed: 随机种子
    """
    hidden_states = tf.keras.Input(shape=(None, None))
    mask = tf.keras.Input(shape=(None, None, None))
    outputs = hidden_states

    for _ in range(config.num_hidden_layers):
        outputs = bert_layer(config=config, is_training=is_training, manual_seed=manual_seed)(outputs, mask)

    return tf.keras.Model(inputs=[hidden_states, mask], output=outputs)


def bert_pooler(config: BertConfig) -> tf.keras.Model:
    """Bert Pooler
    :param config: BertConfig实例
    """
    hidden_states = tf.keras.Input(shape=(None, None))
    if config.segment_type == "relative":
        first_token_tensor = tf.reduce_mean(input_tensor=hidden_states[:, 0:2], axis=1)
    else:
        first_token_tensor = hidden_states[:, 0]

    pooled_output = tf.keras.layers.Dense(units=config.hidden_size, activation="tanh")(first_token_tensor)

    return tf.keras.Model(inputs=hidden_states, outputs=pooled_output)



