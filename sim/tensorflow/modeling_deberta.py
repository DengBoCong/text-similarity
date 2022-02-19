#! -*- coding: utf-8 -*-
""" Tensorflow DeBERTa Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import tensorflow.keras as keras
from sim.tensorflow.layers import PositionEmbedding
from sim.tools import BertConfig
from typing import Any


def bert_embedding(hidden_size: int,
                   embedding_size: int,
                   token_embeddings: Any,
                   hidden_dropout_prob: float = None,
                   shared_segment_embeddings: bool = False,
                   max_position: int = 512,
                   position_merge_mode: str = "add",
                   hierarchical_position: Any = None,
                   segment_vocab_size: int = 2,
                   layer_norm_eps: float = 1e-12,
                   initializer: Any = None,
                   position_ids: Any = None,
                   name: str = "embedding") -> keras.Model:
    """Bert Embedding
    :param hidden_size: 编码维度
    :param embedding_size: 词嵌入大小
    :param token_embeddings: word embedding
    :param hidden_dropout_prob: Dropout比例
    :param shared_segment_embeddings: 若True，则segment跟token共用embedding
    :param max_position: 绝对位置编码最大位置数
    :param position_merge_mode: 输入和position合并的方式
    :param hierarchical_position: 是否层次分解位置编码
    :param segment_vocab_size: segment总数目
    :param layer_norm_eps: layer norm 附加因子，避免除零
    :param initializer: Embedding的初始化器
    :param position_ids: 位置编码ids
    :param name: 模型名
    """
    input_ids = keras.Input(shape=(None,))
    segment_ids = keras.Input(shape=(None,))

    # 默认使用截断正态分布初始化
    if not initializer:
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)

    outputs = token_embeddings(input_ids)

    if segment_vocab_size > 0:
        if shared_segment_embeddings:
            segment_embeddings = token_embeddings(segment_ids)
        else:
            segment_embeddings = keras.layers.Embedding(
                input_dim=segment_vocab_size,
                output_dim=embedding_size,
                embeddings_initializer=initializer,
                name=f"{name}-segment"
            )(segment_ids)

        outputs = keras.layers.Add(name=f"{name}-token-segment")([outputs, segment_embeddings])

    position_embeddings = PositionEmbedding(
        input_dim=max_position,
        output_dim=embedding_size,
        merge_mode=position_merge_mode,
        hierarchical=hierarchical_position,
        custom_position_ids=position_ids is not None,
        embeddings_initializer=initializer,
        name=f"{name}-position"
    )
    if position_ids is None:
        outputs = position_embeddings(outputs)
    else:
        outputs = position_embeddings([outputs, position_ids])

    outputs = keras.layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}-norm").compute_mask(outputs)
    outputs = keras.layers.Dropout(rate=hidden_dropout_prob, name=f"{name}-dropout")(outputs)

    if embedding_size != hidden_size:
        outputs = keras.layers.Dense(
            units=hidden_size,
            kernel_initializer=initializer,
            name=f"{name}-mapping"
        )(outputs)

    return keras.Model(inputs=[input_ids, segment_ids], outputs=outputs, name=name)


def bert_model(config: BertConfig,
               batch_size: int,
               position_merge_mode: str = "add",
               is_training: bool = True,
               add_pooling_layer: bool = True,
               with_pool: Any = False,
               with_nsp: Any = False,
               with_mlm: Any = False,
               name: str = "bert") -> keras.Model:
    """Bert Model
    :param config: BertConfig实例
    :param batch_size: batch size
    :param position_merge_mode: 输入和position合并的方式
    :param is_training: train/eval
    :param add_pooling_layer: 处理输出，后面三个参数用于此
    :param with_pool: 是否包含Pool部分, 必传hidden_size
    :param with_nsp: 是否包含NSP部分
    :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, token_embeddings
    :param name: 模型名
    """
    input_ids = keras.Input(shape=(None,))
    token_type_ids = keras.Input(shape=(None,))
    input_mask = tf.cast(x=tf.math.equal(input_ids, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]
    initializer = keras.initializers.TruncatedNormal(stddev=0.02)

    config = copy.deepcopy(config)
    if not is_training:
        config.hidden_dropout_prob = 0.0
        config.attention_prob_dropout_prob = 0.0