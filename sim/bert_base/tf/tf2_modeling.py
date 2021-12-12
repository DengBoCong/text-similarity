#! -*- coding: utf-8 -*-
""" Implementation of Bert
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sim.bert_base import BertConfig
from typing import NoReturn


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


def bert_embedding(config: BertConfig, is_training: bool, manual_seed: int = 1) -> tf.keras.Model:
    """Bert Embedding
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param manual_seed: 随机种子
    """
    input_ids = tf.keras.Input(shape=(None,))
    token_type_ids = tf.keras.Input(shape=(None,))
    segment_token_type_ids = token_type_ids % 10
    diff_token_type_ids = token_type_ids // 10

    batch_size, seq_len = tf.shape(input_ids)[0], tf.shape(input_ids)[1]

    word_embeddings = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.hidden_size)(input_ids)

    if config.use_relative_position:
        pos_encoding = gen_relative_positions_embeddings(position=config.vocab_size, d_model=config.hidden_size)
        position_embeddings = pos_encoding[:, :seq_len, :]
    else:
        position_ids = tf.expand_dims(input=tf.range(start=0, limit=seq_len, delta=1), axis=0)
        position_ids = tf.repeat(input=position_ids, repeats=[batch_size], axis=0)
        position_embeddings = tf.keras.layers.Embedding(
            input_dim=config.max_position_embeddings, output_dim=config.hidden_size)(position_ids)
    segment_token_type_embeddings = tf.keras.layers.Embedding(
        input_dim=config.type_vocab_size, output_dim=config.hidden_size)(segment_token_type_ids)
    diff_token_type_embeddings = tf.keras.layers.Embedding(
        input_dim=5, output_dim=config.hidden_size)(diff_token_type_ids)
    # token_type_embeddings = segment_token_type_embeddings + diff_token_type_embeddings
    token_type_embeddings = segment_token_type_embeddings

    embeddings = word_embeddings + position_embeddings + token_type_embeddings
    layer_norm_output = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(embeddings)
    dropout_output = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob,
                                             seed=manual_seed)(layer_norm_output, is_training)

    return tf.keras.Model(inputs=[input_ids, token_type_ids], outputs=dropout_output)


class BertModel(object):
    """Bert Model"""

    def __init__(self,
                 config: BertConfig,
                 is_training: bool,
                 input_ids: tf.Tensor,
                 input_mask: tf.Tensor = None,
                 token_type_ids: tf.Tensor = None,
                 use_one_hot_embeddings: bool = False,
                 manual_seed: int = 1,
                 training: bool = True) -> NoReturn:
        """构建BertModel
        :param config: BertConfig实例
        :param is_training: train/eval
        :param input_ids: int32, [batch_size, seq_length]
        :param input_mask: int32, [batch_size, seq_length]
        :param token_type_ids: int32, [batch_size, seq_length]
        :param use_one_hot_embeddings: 是否使用one-hot embedding
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_prob_dropout_prob = 0.0

        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_len], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_len], dtype=tf.int32)

        # 关于embedding这一部分可以直接切换keras api的Embedding层，当然，可以像如下这样使用tensorflow api的习惯
        # self.embedding_output, self.embedding_table = embedding_lookup(
        #     input_ids=input_ids,
        #     vocab_size=config.vocab_size,
        #     embedding_size=config.hidden_size,
        #     initializer_range=config.initializer_range,
        #     use_one_hot_embeddings=use_one_hot_embeddings
        # )
        #
        # self.embedding_output = embedding_postprocessor(
        #     input_tensor=self.embedding_output,
        #     use_token_type=True,
        #     token_type_ids=token_type_ids,
        #     token_type_vocab_size=config.type_vocab_size,
        #     use_position_embeddings=True,
        #     initializer_range=config.initializer_range,
        #     max_position_embeddings=config.max_position_embeddings,
        #     dropout_prob=config.hidden_dropout_prob,
        #     manual_seed=manual_seed,
        #     training=training
        # )
