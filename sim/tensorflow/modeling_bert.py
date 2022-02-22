#! -*- coding: utf-8 -*-
""" Tensorflow Bert Common Modules
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
from sim.tools import BertConfig
from sim.tensorflow.common import recompute_grad
from sim.tensorflow.layers import BertOutput
from sim.tensorflow.layers import BertSelfAttention
from sim.tensorflow.layers import Embedding
from sim.tensorflow.layers import FeedForward
from sim.tensorflow.layers import PositionEmbedding
from typing import Any


def bert_embedding(hidden_size: int,
                   embedding_size: int,
                   token_embeddings: Any,
                   hidden_dropout_prob: float = None,
                   shared_segment_embeddings: bool = False,
                   max_position: int = None,
                   position_merge_mode: str = "add",
                   hierarchical_position: Any = None,
                   type_vocab_size: int = None,
                   layer_norm_eps: float = None,
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

    outputs = keras.layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}-norm")(outputs)
    outputs = keras.layers.Dropout(rate=hidden_dropout_prob, name=f"{name}-dropout")(outputs)

    if embedding_size != hidden_size:
        outputs = keras.layers.Dense(
            units=hidden_size,
            kernel_initializer=initializer,
            name=f"{name}-mapping"
        )(outputs)

    return keras.Model(inputs=[input_ids, segment_ids], outputs=outputs, name=name)


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
        self.initializer = initializer if initializer else keras.initializers.TruncatedNormal(stddev=config.initializer_range)
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

    @recompute_grad
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
    input_mask = tf.cast(x=tf.math.equal(input_ids, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]
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
    )([input_ids, token_type_ids])

    for index in range(config.num_hidden_layers):
        outputs = BertLayer(config=config, batch_size=batch_size, name=f"bert-layer-{index}")([outputs, input_mask])

    if add_pooling_layer:
        argument = {}
        if with_pool:
            argument["hidden_size"] = config.hidden_size
        if with_mlm:
            argument["embedding_size"] = config.embedding_size
            argument["hidden_act"] = config.hidden_act
            argument["layer_norm_eps"] = config.layer_norm_eps
            argument["vocab_dense_layer"] = token_embeddings

        outputs = BertOutput(with_pool, with_nsp, with_mlm, initializer=initializer,
                             name="bert-output", **argument)(outputs)

    return keras.Model(inputs=[input_ids, token_type_ids], outputs=outputs, name=name)
