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





def split_heads(input_tensor: tf.Tensor, head_num: int, head_size: int):
    """分拆最后一个维度到 (num_heads, depth)
    :param input_tensor: 输入
    :param head_num: 注意力头数
    :param head_size: 每个注意力头维数
    """
    batch_size = input_tensor.shape[0]
    input_tensor = tf.reshape(input_tensor, (batch_size, -1, head_num, head_size))
    return tf.transpose(input_tensor, perm=[0, 2, 1, 3])


def scaled_dot_product_attention(query: tf.Tensor,
                                 key: tf.Tensor,
                                 value: tf.Tensor,
                                 hidden_size: int,
                                 dropout: float,
                                 is_training: bool,
                                 mask: Any = None,
                                 manual_seed: int = 1) -> tuple:
    """点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param hidden_size: hidden size
    :param dropout: 注意力dropout
    :param is_training: 是否处于训练模式
    :param mask: float, (..., seq_len_q, seq_len_k)
    :param manual_seed: 随机种子
    """
    batch_size = tf.shape(query)[0]
    attention_scores = tf.matmul(a=query, b=key, transpose_b=True)
    dk = tf.cast(x=tf.shape(input=k)[-1], dtype=tf.float32)
    attention_scores = attention_scores / tf.math.sqrt(x=dk)

    if mask is not None:
        attention_scores += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits=attention_scores, axis=-1)
    attention_weights = tf.keras.layers.Dropout(rate=dropout, seed=manual_seed)(attention_weights, is_training)

    context_layer = tf.matmul(a=attention_weights, b=value)
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
    context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, hidden_size))

    return context_layer, attention_weights


def bert_self_attention(config: BertConfig, is_training: bool, manual_seed: int = 1) -> tf.keras.Model:
    """Bert Self-Attention
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param manual_seed: 随机种子
    """
    hidden_states = tf.keras.Input(shape=(None, None))
    mask = tf.keras.Input(shape=(None, None, None))

    assert config.hidden_size % config.num_attention_heads == 0
    attention_head_size = config.hidden_size // config.num_attention_heads

    query = tf.keras.layers.Dense(units=config.hidden_size)(hidden_states)
    key = tf.keras.layers.Dense(units=config.hidden_size)(hidden_states)
    value = tf.keras.layers.Dense(units=config.hidden_size)(hidden_states)

    query = split_heads(input_tensor=query, head_num=config.num_attention_heads, head_size=attention_head_size)
    key = split_heads(input_tensor=key, head_num=config.num_attention_heads, head_size=attention_head_size)
    value = split_heads(input_tensor=value, head_num=config.num_attention_heads, head_size=attention_head_size)

    scaled_attention, attention_weights = scaled_dot_product_attention(
        query=query, key=key, value=value, hidden_size=config.hidden_size, dropout=config.attention_prob_dropout_prob,
        is_training=is_training, mask=mask, manual_seed=manual_seed)

    return tf.keras.Model(inputs=[hidden_states, mask], outputs=[scaled_attention, attention_weights])


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


def bert_attention(config: BertConfig, is_training: bool, manual_seed: int = 1) -> tf.keras.Model:
    """Bert Attention
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param manual_seed: 随机种子
    """
    hidden_states = tf.keras.Input(shape=(None, None))
    mask = tf.keras.Input(shape=(None, None, None))

    self_outputs = bert_self_attention(config=config, is_training=is_training,
                                       manual_seed=manual_seed)(hidden_states, mask)
    attention_output = bert_self_output(config=config, is_training=is_training,
                                        manual_seed=manual_seed)(self_outputs[0], hidden_states)

    return tf.keras.Model(inputs=[hidden_states, mask], outputs=[attention_output, self_outputs[1]])


def bert_layer(config: BertConfig, is_training: bool, manual_seed: int = 1) -> tf.keras.Model:
    """Bert Layer
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param manual_seed: 随机种子
    """
    hidden_states = tf.keras.Input(shape=(None, None))
    mask = tf.keras.Input(shape=(None, None, None))

    attention_output, attention_weights = bert_attention(config=config, is_training=is_training,
                                                         manual_seed=manual_seed)(hidden_states, mask)
    outputs = tf.keras.layers.Dense(units=config.intermediate_size, activation=config.hidden_act)(attention_output)
    outputs = bert_self_output(config=config, is_training=is_training,
                               manual_seed=manual_seed)(outputs, attention_output)

    return tf.keras.Model(inputs=[hidden_states, mask], outputs=outputs)


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


def bert_model(config: BertConfig,
               is_training: bool,
               add_pooling_layer: bool = True,
               manual_seed: int = 1) -> tf.keras.Model:
    """Bert Model
    :param config: BertConfig实例
    :param is_training: train/eval
    :param add_pooling_layer: 添加池化层
    :param manual_seed: 随机种子
    """
    input_ids = tf.keras.Input(shape=(None,))
    token_type_ids = tf.keras.Input(shape=(None,))
    input_mask = tf.cast(x=tf.math.equal(input_ids, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]

    config = copy.deepcopy(config)
    if not is_training:
        config.hidden_dropout_prob = 0.0
        config.attention_prob_dropout_prob = 0.0

    embedding_output = bert_embedding(config=config, is_training=is_training,
                                      manual_seed=manual_seed)(input_ids, token_type_ids)
    encoder_output = bert_encoder(config=config, is_training=is_training,
                                  manual_seed=manual_seed)(embedding_output, input_mask)
    if add_pooling_layer and not config.use_mean_pooling:
        pooler_output = bert_pooler(config=config)(encoder_output)
    elif config.use_mean_pooling:
        mask = tf.cast(x=tf.math.not_equal(x=input_ids, y=0), dtype=tf.float32)
        sum_mask = tf.reduce_sum(input_tensor=mask, axis=-1, keepdims=True)
        mul_msk = tf.reduce_sum(input_tensor=tf.multiply(x=tf.expand_dims(input=mask, axis=-1),
                                                         y=encoder_output), axis=1)
        pooler_output = tf.divide(x=mul_msk, y=sum_mask)
    else:
        pooler_output = None

    return tf.keras.Model(inputs=[input_ids, input_mask], outputs=[encoder_output, pooler_output])
