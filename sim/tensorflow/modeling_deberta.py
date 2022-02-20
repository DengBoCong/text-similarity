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
import math

import tensorflow as tf
import tensorflow.keras as keras
from sim.tensorflow.layers import Embedding
from sim.tensorflow.layers import PositionEmbedding
from sim.tools import BertConfig
from sim.tools.tools import build_relative_position_deberta
from typing import Any


def bert_embedding(hidden_size: int,
                   embedding_size: int,
                   token_embeddings: Any,
                   hidden_dropout_prob: float = None,
                   shared_segment_embeddings: bool = False,
                   max_position: int = 512,
                   position_merge_mode: str = "add",
                   hierarchical_position: Any = None,
                   type_vocab_size: int = 2,
                   position_biased_input: bool = False,
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
    :param type_vocab_size: segment总数目
    :param position_biased_input: 是否增加位置embedding
    :param layer_norm_eps: layer norm 附加因子，避免除零
    :param initializer: Embedding的初始化器
    :param position_ids: 位置编码ids
    :param name: 模型名
    """
    input_ids = keras.Input(shape=(None,))
    segment_ids = keras.Input(shape=(None,))
    inputs_mask = keras.Input(shape=(None, 1))

    # 默认使用截断正态分布初始化
    if not initializer:
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)

    outputs = token_embeddings(input_ids)

    if type_vocab_size > 0:
        if shared_segment_embeddings:
            segment_embeddings = token_embeddings(segment_ids)
        else:
            segment_embeddings = keras.layers.Embedding(
                input_dim=type_vocab_size,
                output_dim=embedding_size,
                embeddings_initializer=initializer,
                name=f"{name}-segment"
            )(segment_ids)

        outputs = keras.layers.Add(name=f"{name}-token-segment")([outputs, segment_embeddings])

    if position_biased_input:
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

    if embedding_size != hidden_size:
        outputs = keras.layers.Dense(
            units=hidden_size,
            kernel_initializer=initializer,
            use_bias=False,
            name=f"{name}-mapping"
        )(outputs)

    outputs = keras.layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}-norm")(outputs)
    outputs = inputs_mask * outputs
    outputs = keras.layers.Dropout(rate=hidden_dropout_prob, name=f"{name}-dropout")(outputs)

    return keras.Model(inputs=[input_ids, segment_ids], outputs=outputs, name=name)


class DisentangledSelfAttention(keras.layers.Layer):
    """分散注意力机制
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 batch_size: int,
                 attention_dropout: float,
                 hidden_dropout_prob: float,
                 pos_ebd_size: int,
                 use_bias: bool = True,
                 key_size: int = None,
                 hidden_size: int = None,
                 initializer: Any = "glorot_uniform",
                 pos_type: str = None,

                 position_buckets: int = -1,
                 max_relative_positions: int = -1,
                 share_att_key: bool = False,
                 pos_att_type: str = "c2p|p2c",
                 **kwargs):
        """
        :param num_heads: 注意力头数
        :param head_size: Attention中V的head_size
        :param batch_size: batch size
        :param attention_dropout: Attention矩阵的Dropout比例
        :param hidden_dropout_prob: 加一层dropout
        :param pos_ebd_size: 位置编码大小
        :param use_bias: 是否加上偏差项
        :param key_size: Attention中Q,K的head_size
        :param hidden_size: 编码维度
        :param initializer: 初始化器
        :param pos_type: 指定位置编码种类，现支持经典的相对位置编码: "typical_relation"
        :param position_buckets: bucket size
        :param max_relative_positions: 最大位置编码
        :param share_att_key: 共享key
        :param pos_att_type: 注意力类型
        """
        super(DisentangledSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.batch_size = batch_size
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pos_ebd_size = pos_ebd_size
        self.use_bias = use_bias
        self.key_size = key_size if key_size is not None else head_size
        self.hidden_size = hidden_size if hidden_size is not None else num_heads * head_size
        self.initializer = initializer
        self.pos_type = pos_type

        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.share_att_key = share_att_key
        self.pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')]

    def build(self, input_shape):
        super(DisentangledSelfAttention, self).build(input_shape)
        self.query_dense = keras.layers.Dense(units=self.key_size * self.num_heads, use_bias=self.use_bias,
                                              kernel_initializer=self.initializer, name="query")
        self.key_dense = keras.layers.Dense(units=self.key_size * self.num_heads, use_bias=self.use_bias,
                                            kernel_initializer=self.initializer, name="key")
        self.value_dense = keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=self.use_bias,
                                              kernel_initializer=self.initializer, name="value")
        self.pos_dropout = keras.layers.Dropout(rate=self.hidden_dropout_prob)

        if not self.share_att_key:
            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_key_dense = keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=self.use_bias,
                                                        kernel_initializer=self.initializer, name="pos-key")
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_query_dense = keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=self.use_bias,
                                                          kernel_initializer=self.initializer, name="pos-query")

        self.dropout = keras.layers.Dropout(rate=self.attention_dropout)

    def transpose_for_scores(self, input_tensor: tf.Tensor, head_size: int):
        """分拆最后一个维度到 (num_heads, depth)
        :param input_tensor: 输入
        :param head_size: 每个注意力头维数
        """
        input_tensor = tf.reshape(tensor=input_tensor, shape=(self.batch_size, -1, self.num_heads, head_size))
        return tf.transpose(input_tensor, perm=[0, 2, 1, 3])

    def call(self, inputs, *args, **kwargs):
        query, key, value, mask = inputs

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.transpose_for_scores(input_tensor=query, head_size=self.key_size)
        key = self.transpose_for_scores(input_tensor=key, head_size=self.key_size)
        value = self.transpose_for_scores(input_tensor=value, head_size=self.head_size)

        rel_att, scale_factor = None, 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        if "p2p" in self.pos_att_type:
            scale_factor += 1
        scale = 1 / tf.sqrt(query.shape[-1] * scale_factor)
        attention_scores = 




        return attn_outputs, attention_weights

    def get_config(self):
        config = {
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "attention_dropout": self.attention_dropout,
            "use_bias": self.use_bias,
            "key_size": self.key_size,
            "hidden_size": self.hidden_size,
            "initializer": keras.initializers.serialize(initializer=self.initializer),
        }
        base_config = super(DisentangledSelfAttention, self).get_config()
        base_config.update(config)
        return base_config


class BertLayer(keras.layers.Layer):
    """bert block
    """

    def __init__(self, config: BertConfig, batch_size: int, initializer: Any = None, **kwargs):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param initializer: 初始化器
        """
        super(BertLayer, self).__init__(**kwargs)
        self.bert_config = config
        self.batch_size = batch_size
        self.initializer = initializer if initializer else keras.initializers.TruncatedNormal(
            stddev=config.initializer_range)
        self.attn_name = "multi-head-self-attention"
        self.feed_forward_name = "feedforward"

    def build(self, input_shape):
        super(BertLayer, self).build(input_shape)
        self.bert_self_attention = BertSelfAttention(
            num_heads=self.bert_config.num_attention_heads,
            head_size=self.bert_config.attention_head_size,
            batch_size=self.batch_size,
            attention_dropout=self.bert_config.attention_probs_dropout_prob,
            key_size=self.bert_config.attention_key_size,
            hidden_size=self.bert_config.hidden_size,
            initializer=self.initializer,
            name=self.attn_name
        )
        self.attn_dropout = keras.layers.Dropout(rate=self.bert_config.hidden_dropout_prob,
                                                 name=f"{self.attn_name}-dropout")
        self.attn_add = keras.layers.Add(name=f"{self.attn_name}-add")
        self.attn_norm = keras.layers.LayerNormalization(epsilon=self.bert_config.layer_norm_eps,
                                                         name=f"{self.attn_name}-norm")

        self.feedforward = FeedForward(units=self.bert_config.intermediate_size, activation=self.bert_config.hidden_act,
                                       kernel_initializer=self.initializer, name=self.feed_forward_name)

        self.feedforward_dropout = keras.layers.Dropout(rate=self.bert_config.hidden_dropout_prob,
                                                        name=f"{self.feed_forward_name}-dropout")
        self.feedforward_add = keras.layers.Add(name=f"{self.feed_forward_name}-add")
        self.feedforward_norm = keras.layers.LayerNormalization(epsilon=self.bert_config.layer_norm_eps,
                                                                name=f"{self.feed_forward_name}-norm")

    def call(self, inputs, *args, **kwargs):
        inputs, mask = inputs
        attn_outputs, attn_weights = self.bert_self_attention([inputs, inputs, inputs, mask])
        attn_outputs = self.attn_dropout(attn_outputs)
        attn_outputs = self.attn_add([attn_outputs, inputs])
        attn_outputs = self.attn_norm(attn_outputs)

        outputs = self.feedforward(attn_outputs)
        outputs = self.feedforward_dropout(outputs)
        outputs = self.feedforward_add([outputs, attn_outputs])
        outputs = self.feedforward_norm(outputs)

        return outputs


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
    input_mask = tf.cast(x=tf.math.equal(input_ids, 0), dtype=tf.float32)
    initializer = keras.initializers.TruncatedNormal(stddev=config.initializer_range)

    config = copy.deepcopy(config)
    if not is_training:
        config.hidden_dropout_prob = 0.0
        config.attention_prob_dropout_prob = 0.0

    token_embeddings = Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_size,
        embeddings_initializer=initializer,
        mask_zero=True,
        name="embedding-token"
    )

    outputs = bert_embedding(
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
        token_embeddings=token_embeddings,
        hidden_dropout_prob=config.hidden_dropout_prob,
        shared_segment_embeddings=config.shared_segment_embeddings,
        max_position=config.max_position,
        position_merge_mode=position_merge_mode,
        hierarchical_position=config.hierarchical_position,
        type_vocab_size=config.type_vocab_size,
        layer_norm_eps=config.layer_norm_eps
    )([input_ids, token_type_ids, input_mask[:, :, tf.newaxis]])

    max_relative_positions = getattr(config, 'max_relative_positions', -1)
    if max_relative_positions < 1:
        max_relative_positions = config.max_position_embeddings
    position_buckets = getattr(config, 'position_buckets', -1)
    pos_ebd_size = max_relative_positions * 2
    if position_buckets > 0:
        pos_ebd_size = position_buckets * 2
    rel_embeddings = keras.layers.Embedding(input_dim=pos_ebd_size, output_dim=config.hidden_size)

    relative_pos = build_relative_position_deberta(
        query_size=outputs.shape[1],
        key_size=outputs.shape[1],
        bucket_size=position_buckets,
        max_position=max_relative_positions
    )

    rel_embedding = rel_embeddings.embeddings
    norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]
    if rel_embedding is not None and ("layer_norm" in norm_rel_ebd):
        rel_embedding = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(rel_embedding)

    for index in range(config.num_hidden_layers):
        outputs = BertLayer(config=config, batch_size=batch_size, name=f"bert-layer-{index}")([outputs, input_mask])
