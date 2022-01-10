#! -*- coding: utf-8 -*-
""" Tensorflow Bert Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras
from sim.tools import BertConfig
from typing import Any


class PositionEmbedding(keras.layers.Layer):
    """定义可训练的位置Embedding
    """

    def __init__(self,
                 config: BertConfig,
                 merge_mode: str = "add",
                 hierarchical: Any = None,
                 embeddings_initializer: Any = "zeros",
                 **kwargs):
        """
        :param config: BertConfig实例
        :param merge_mode: 输入和position合并的方式
        :param hierarchical: 是否层次分解位置编码
        :param embeddings_initializer: 初始化器
        """
        super(PositionEmbedding, self).__init__(**kwargs)
        self.config = config
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = embeddings_initializer
        self.embeddings = None
        self.custom_position_ids = False

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.config.max_position, self.config.embedding_size),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs, *args, **kwargs):
        """如果传入自定义position_ids，那么第二个输入为自定义的位置id
        """
        assert len(inputs) <= 2
        if len(inputs) == 2:
            self.custom_position_ids = True
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
            embeddings_x = tf.gather(params=embeddings, indices=position_ids // self.config.max_position)
            embeddings_y = tf.gather(params=embeddings, indices=position_ids % self.config.max_position)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if len(inputs) == 2:
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
            if len(inputs) != 2:
                embeddings = tf.tile(input=embeddings, multiples=[batch_size, 1, 1])
            return tf.concat(values=[inputs, embeddings], axis=-1)

    def get_config(self):
        config = {
            "max_position": self.config.max_position,
            "embedding_size": self.config.embedding_size,
            "merge_model": self.merge_mode,
            "hierarchical": self.hierarchical,
            "embeddings_initializer": keras.initializers.serialize(initializer=self.embeddings_initializer),
            "custom_position_ids": self.custom_position_ids
        }
        base_config = super(PositionEmbedding, self).get_config()
        base_config.update(config)
        return base_config


def bert_embedding(config: BertConfig, is_training: bool, manual_seed: int = 1,
                   initializer: Any = None, position_ids: Any = None) -> keras.Model:
    """Bert Embedding
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param manual_seed: 随机种子
    :param initializer: Embedding的初始化器
    :param position_ids: 位置编码ids
    """
    input_ids = keras.Input(shape=(None,))
    segment_ids = keras.Input(shape=(None,))

    # 默认使用截断正态分布初始化
    if not initializer:
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)

    word_embeddings = keras.layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_size,
        embeddings_initializer=initializer,
        mask_zero=True,
        name="embedding-token"
    )
    outputs = word_embeddings(input_ids)

    if config.segment_vocab_size > 0:
        if config.shared_segment_embeddings:
            segment_embeddings = word_embeddings(segment_ids)
        else:
            segment_embeddings = keras.layers.Embedding(
                input_dim=config.segment_vocab_size,
                output_dim=config.embedding_size,
                embeddings_initializer=initializer,
                name="embedding-segment"
            )(segment_ids)

        outputs = keras.layers.Add(name="embedding-token-segment")([outputs, segment_embeddings])

    position_embeddings = PositionEmbedding(
        config=config,
        hierarchical=config.hierarchical_position,
        embeddings_initializer=initializer,
        name="embedding-position"
    )
    if position_ids is None:
        outputs = position_embeddings(outputs)
    else:
        outputs = position_embeddings([outputs, position_ids])

    outputs = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="embedding-norm")(outputs)
    outputs = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob,
                                      seed=manual_seed, name="embedding-dropout")(outputs, is_training)

    if config.embedding_size != config.hidden_size:
        outputs = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=initializer,
            name="embedding-mapping"
        )(outputs)

    return outputs



