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
from sim.tensorflow.layers import BertOutput
from sim.tensorflow.layers import Embedding
from sim.tensorflow.layers import FeedForward
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

    return keras.Model(inputs=[input_ids, segment_ids, inputs_mask], outputs=outputs, name=name)


class DisentangledSelfAttention(keras.layers.Layer):
    """分散注意力机制
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 batch_size: int,
                 attention_dropout: float,
                 hidden_dropout_prob: float,
                 use_bias: bool = True,
                 key_size: int = None,
                 hidden_size: int = None,
                 initializer: Any = "glorot_uniform",
                 relative_pos: Any = None,
                 position_buckets: int = -1,
                 max_relative_positions: int = -1,
                 pos_att_type: str = "c2p|p2c",
                 **kwargs):
        """
        :param num_heads: 注意力头数
        :param head_size: Attention中V的head_size
        :param attention_dropout: Attention矩阵的Dropout比例
        :param hidden_dropout_prob: 加一层dropout
        :param pos_ebd_size: 位置编码大小
        :param use_bias: 是否加上偏差项
        :param key_size: Attention中Q,K的head_size
        :param hidden_size: 编码维度
        :param initializer: 初始化器
        :param relative_pos: 相对位置编码
        :param position_buckets: bucket size
        :param max_relative_positions: 最大位置编码
        :param pos_att_type: 注意力类型
        """
        super(DisentangledSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.batch_size = batch_size
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias
        self.key_size = key_size if key_size is not None else head_size
        self.hidden_size = hidden_size if hidden_size is not None else num_heads * head_size
        self.initializer = initializer
        self.relative_pos = relative_pos
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')]
        self.pos_ebd_size = position_buckets if position_buckets > 0 else max_relative_positions

    def build(self, input_shape):
        super(DisentangledSelfAttention, self).build(input_shape)
        self.query_dense = keras.layers.Dense(units=self.key_size * self.num_heads, use_bias=self.use_bias,
                                              kernel_initializer=self.initializer, name="query")
        self.key_dense = keras.layers.Dense(units=self.key_size * self.num_heads, use_bias=self.use_bias,
                                            kernel_initializer=self.initializer, name="key")
        self.value_dense = keras.layers.Dense(units=self.head_size * self.num_heads, use_bias=self.use_bias,
                                              kernel_initializer=self.initializer, name="value")
        self.pos_dropout = keras.layers.Dropout(rate=self.hidden_dropout_prob)

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
        query, key, value, rel_embeddings, mask = inputs

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.transpose_for_scores(input_tensor=query, head_size=self.key_size)
        key = self.transpose_for_scores(input_tensor=key, head_size=self.key_size)
        value = self.transpose_for_scores(input_tensor=value, head_size=self.head_size)

        rel_att, scale_factor = None, 1.
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        if "p2p" in self.pos_att_type:
            scale_factor += 1
        scale = 1 / tf.sqrt(query.shape[-1] * scale_factor)
        attention_scores = tf.matmul(a=query, b=key * scale, transpose_b=True)

        rel_embeddings = self.pos_dropout(rel_embeddings)
        rel_att = self.disentangled_attention_bias(query=query, key=key, relative_pos=self.relative_pos,
                                                   rel_embeddings=rel_embeddings, scale_factor=scale_factor)
        attention_scores = attention_scores + rel_att

        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits=attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        context_layer = tf.matmul(a=attention_weights, b=value)
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(tensor=context_layer, shape=(self.batch_size, -1, self.head_size * self.num_heads))

        return context_layer, attention_weights

    def disentangled_attention_bias(self, query, key, relative_pos, rel_embeddings, scale_factor):
        """分散注意力机制计算
        :param query:
        :param key:
        :param relative_pos: 相对位置编码
        :param rel_embeddings:
        :param scale_factor: 缩放因子
        """
        if len(relative_pos.shape) == 2:
            relative_pos = relative_pos[tf.newaxis, tf.newaxis, :, :]
        elif len(relative_pos.shape) == 3:
            relative_pos = tf.expand_dims(input=relative_pos, axis=1)
        elif len(relative_pos.shape) != 4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.shape}')

        att_span = self.pos_ebd_size
        rel_embeddings = tf.expand_dims(rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :], 0)

        if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
            # pos_key_layer = self.transpose_for_scores(self.pos_key_dense(rel_embeddings), self.num_heads)
            pos_key_layer = self.pos_key_dense(rel_embeddings)
            pos_key_layer = tf.reshape(tensor=pos_key_layer, shape=(pos_key_layer.shape[0],
                                                                    pos_key_layer.shape[1], self.num_heads, -1))
            pos_key_layer = tf.transpose(pos_key_layer, perm=[0, 2, 1, 3])
            print((query.shape[0] // self.num_heads, 1, 1))
            exit(0)(1, 12, 1024, 64),
            pos_key_layer = tf.repeat(pos_key_layer, repeats=(query.shape[0] // self.num_heads, 1, 1))
            print(pos_key_layer)
            exit(0)
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer = self.transpose_for_scores(self.pos_query_dense(rel_embeddings), self.num_heads)
            pos_query_layer = tf.repeat(pos_query_layer, repeats=(query.shape[0] // self.num_heads, 1, 1))

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = 1 / math.sqrt(pos_key_layer.shape[-1] * scale_factor)
            c2p_att = tf.matmul(a=query, b=pos_key_layer * scale, transpose_b=True)
            c2p_pos = tf.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = tf.gather(c2p_att, indices=c2p_pos)
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            scale = 1 / math.sqrt(pos_query_layer.shape[-1] * scale_factor)
            if key.shape[-2] != query.shape[-2]:
                r_pos = build_relative_position_deberta(key.shape[-2], key.shape[-2],
                                                        bucket_size=self.position_buckets,
                                                        max_position=self.max_relative_positions)
                r_pos = tf.expand_dims(r_pos, 0)
            else:
                r_pos = relative_pos

            p2c_pos = tf.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
            if key.shape[-2] != query.shape[-2]:
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if "p2c" in self.pos_att_type:
            p2c_att = tf.matmul(a=key, b=pos_query_layer * scale, transpose_b=True)
            p2c_att = tf.gather(p2c_att, indices=p2c_pos)

            if key.shape[-2] != query.shape[-2]:
                p2c_att = tf.gather(p2c_att, indices=pos_index)
            score += p2c_att

        # position->position
        if "p2p" in self.pos_att_type:
            pos_query = pos_query_layer[:, :, att_span:, :]
            p2p_att = tf.matmul(a=pos_query, b=pos_key_layer, transpose_b=True)
            if key.shape[-2] != query.shape[-2]:
                p2p_att = tf.gather(p2p_att, indices=pos_index)
            p2p_att = tf.gather(p2p_att, indices=c2p_pos)
            score += p2p_att

        return score

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

    def __init__(self,
                 config: BertConfig,
                 batch_size: int,
                 relative_pos: Any = None,
                 bucket_size: int = -1,
                 max_position: int = -1,
                 initializer: Any = None,
                 pos_att_type: str = "c2p|p2c",
                 **kwargs):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param relative_pos: 位置编码
        :param bucket_size: bucket size, 如果relative_pos为空必传
        :param max_position: 最大位置, 如果relative_pos为空必传
        :param initializer: 初始化器
        :param pos_att_type: 注意力类型
        """
        super(BertLayer, self).__init__(**kwargs)
        self.bert_config = config
        self.batch_size = batch_size
        self.relative_pos = relative_pos
        self.bucket_size = bucket_size
        self.max_position = max_position
        self.initializer = initializer if initializer else keras.initializers.TruncatedNormal(
            stddev=config.initializer_range)
        self.pos_att_type = pos_att_type
        self.attn_name = "multi-head-self-attention"
        self.feed_forward_name = "feedforward"

    def build(self, input_shape):
        super(BertLayer, self).build(input_shape)
        if self.relative_pos is None:
            self.relative_pos = build_relative_position_deberta(
                query_size=input_shape[0][1],
                key_size=input_shape[0][1],
                bucket_size=self.bucket_size,
                max_position=self.max_position
            )

        self.disentangled_self_attention = DisentangledSelfAttention(
            num_heads=self.bert_config.num_attention_heads,
            head_size=self.bert_config.attention_head_size,
            batch_size=self.batch_size,
            attention_dropout=self.bert_config.attention_probs_dropout_prob,
            hidden_dropout_prob=self.bert_config.hidden_dropout_prob,
            key_size=self.bert_config.attention_key_size,
            hidden_size=self.bert_config.hidden_size,
            initializer=self.initializer,
            relative_pos=self.relative_pos,
            position_buckets=self.bucket_size,
            max_relative_positions=self.max_position,
            pos_att_type=self.pos_att_type
        )

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
        inputs, rel_embeddings, mask = inputs

        attn_outputs, attn_weights = self.disentangled_self_attention([inputs, inputs, inputs, rel_embeddings, mask])
        attn_outputs = self.attn_norm(attn_outputs)

        outputs = self.feedforward(attn_outputs)
        outputs = self.feedforward_dropout(outputs)
        outputs = self.feedforward_add([outputs, attn_outputs])
        outputs = self.feedforward_norm(outputs)

        return outputs

    def get_config(self):
        config = {
            "config": self.config,
            "batch_size": self.batch_size,
            "bucket_size": self.bucket_size,
            "max_position": self.max_position,
            "initializer": keras.initializers.serialize(initializer=self.initializer),
            "pos_att_type": self.pos_att_type
        }
        base_config = super(BertLayer, self).get_config()
        base_config.update(config)
        return base_config


class DeBERTa(keras.layers.Layer):
    """DeBERTa Model
    """

    def __init__(self,
                 config: BertConfig,
                 batch_size: int,
                 position_merge_mode: str = "add",
                 is_training: bool = True,
                 add_pooling_layer: bool = True,
                 with_pool: Any = False,
                 with_nsp: Any = False,
                 with_mlm: Any = False,
                 name: str = "deberta",
                 **kwargs):
        """
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
        super(DeBERTa, self).__init__(name=name, **kwargs)
        self.config = copy.deepcopy(config)
        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_prob_dropout_prob = 0.0
        self.batch_size = batch_size
        self.position_merge_mode = position_merge_mode
        self.add_polling_layer = add_pooling_layer
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.initializer = keras.initializers.TruncatedNormal(stddev=config.initializer_range)

        self.position_biased_input = getattr(self.config, "position_biased_input", False)
        self.max_relative_positions = getattr(self.config, "max_relative_positions", -1)
        self.pos_att_type = getattr(self.config, "pos_att_type", "c2p|p2c")
        if self.max_relative_positions < 1:
            self.max_relative_positions = self.config.max_position_embeddings
        self.position_buckets = getattr(self.config, 'position_buckets', -1)
        self.pos_ebd_size = self.max_relative_positions * 2
        if self.position_buckets > 0:
            self.pos_ebd_size = self.position_buckets * 2

    def build(self, input_shape):
        super(DeBERTa, self).build(input_shape)

        self.token_embeddings = Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name="embedding-token"
        )

        self.rel_embeddings = self.add_weight(
            name="embedding-rel",
            shape=(self.pos_ebd_size, self.config.hidden_size),
            initializer=self.initializer
        )

    def call(self, inputs, *args, **kwargs):
        input_ids, token_type_ids = inputs
        input_mask = tf.cast(x=tf.math.equal(input_ids, 0), dtype=tf.float32)

        outputs = bert_embedding(
            hidden_size=self.config.hidden_size,
            embedding_size=self.config.embedding_size,
            token_embeddings=self.token_embeddings,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            shared_segment_embeddings=self.config.shared_segment_embeddings,
            max_position=self.config.max_position,
            position_merge_mode=self.position_merge_mode,
            hierarchical_position=self.config.hierarchical_position,
            type_vocab_size=self.config.type_vocab_size,
            position_biased_input=self.position_biased_input,
            layer_norm_eps=self.config.layer_norm_eps
        )([input_ids, token_type_ids, input_mask[:, :, tf.newaxis]])

        rel_embeddings = self.rel_embeddings
        norm_rel_ebd = [x.strip() for x in getattr(self.config, "norm_rel_ebd", "none").lower().split("|")]
        if "layer_norm" in norm_rel_ebd:
            rel_embeddings = keras.layers.LayerNormalization(epsilon=self.config.layer_norm_eps)(rel_embeddings)

        for index in range(self.config.num_hidden_layers):
            outputs = BertLayer(
                config=self.config,
                batch_size=self.batch_size,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                pos_att_type=self.pos_att_type,
                name=f"bert-layer-{index}"
            )([outputs, rel_embeddings, input_mask])

        if self.add_pooling_layer:
            pass
        return outputs

    def get_config(self):
        config = {
            "config": self.config,
            "batch_size": self.batch_size,
            "position_merge_mode": self.position_merge_mode,
            "is_training": self.is_training,
            "add_pooling_layer": self.add_pooling_layer,
            "with_pool": self.with_pool,
            "with_nsp": self.with_nsp,
            "with_mlm": self.with_mlm,
            "initializer": keras.initializers.serialize(initializer=self.initializer),
            "position_biased_input": self.position_biased_input,
            "max_relative_positions": self.max_relative_positions,
            "pos_att_type": self.pos_att_type,
            "position_buckets": self.position_buckets,
            "pos_ebd_size": self.pos_ebd_size
        }
        base_config = super(DeBERTa, self).get_config()
        base_config.update(config)
        return base_config

# def DeBERTa(config: BertConfig,
#             batch_size: int,
#             position_merge_mode: str = "add",
#             is_training: bool = True,
#             add_pooling_layer: bool = True,
#             with_pool: Any = False,
#             with_nsp: Any = False,
#             with_mlm: Any = False,
#             name: str = "deberta") -> keras.Model:
#     """Bert Model
#
#     """
#     input_ids = keras.Input(shape=(None,))
#     token_type_ids = keras.Input(shape=(None,))
#     input_mask = tf.cast(x=tf.math.equal(input_ids, 0), dtype=tf.float32)
#     initializer = keras.initializers.TruncatedNormal(stddev=config.initializer_range)
#
#     config = copy.deepcopy(config)
#     if not is_training:
#         config.hidden_dropout_prob = 0.0
#         config.attention_prob_dropout_prob = 0.0
#
#     token_embeddings = Embedding(
#         input_dim=config.vocab_size,
#         output_dim=config.embedding_size,
#         embeddings_initializer=initializer,
#         mask_zero=True,
#         name="embedding-token"
#     )
#
#     position_biased_input = getattr(config, "position_biased_input", False)
#     outputs = bert_embedding(
#         hidden_size=config.hidden_size,
#         embedding_size=config.embedding_size,
#         token_embeddings=token_embeddings,
#         hidden_dropout_prob=config.hidden_dropout_prob,
#         shared_segment_embeddings=config.shared_segment_embeddings,
#         max_position=config.max_position,
#         position_merge_mode=position_merge_mode,
#         hierarchical_position=config.hierarchical_position,
#         type_vocab_size=config.type_vocab_size,
#         position_biased_input=position_biased_input,
#         layer_norm_eps=config.layer_norm_eps
#     )([input_ids, token_type_ids, input_mask[:, :, tf.newaxis]])
#
#     max_relative_positions = getattr(config, "max_relative_positions", -1)
#     pos_att_type = getattr(config, "pos_att_type", "c2p|p2c")
#     if max_relative_positions < 1:
#         max_relative_positions = config.max_position_embeddings
#     position_buckets = getattr(config, 'position_buckets', -1)
#     pos_ebd_size = max_relative_positions * 2
#     if position_buckets > 0:
#         pos_ebd_size = position_buckets * 2
#
#     rel_embeddings = keras.layers.Embedding(input_dim=pos_ebd_size, output_dim=config.hidden_size)
#     rel_embedding = rel_embeddings.embeddings
#
#
#
#     norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]
#     if rel_embedding is not None and ("layer_norm" in norm_rel_ebd):
#         rel_embedding = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)(rel_embedding)
#
#     for index in range(config.num_hidden_layers):
#         outputs = BertLayer(
#             config=config,
#             batch_size=batch_size,
#             bucket_size=position_buckets,
#             max_position=max_relative_positions,
#             pos_att_type=pos_att_type,
#             name=f"bert-layer-{index}"
#         )([outputs, rel_embedding, input_mask])
#
#     if add_pooling_layer:
#         argument = {}
#         if with_pool:
#             argument["hidden_size"] = config.hidden_size
#         if with_mlm:
#             argument["embedding_size"] = config.embedding_size
#             argument["hidden_act"] = config.hidden_act
#             argument["layer_norm_eps"] = config.layer_norm_eps
#             argument["vocab_dense_layer"] = token_embeddings
#
#         outputs = BertOutput(with_pool, with_nsp, with_mlm, initializer=initializer,
#                              name="bert-output", **argument)(outputs)
#
#     return keras.Model(inputs=[input_ids, token_type_ids], outputs=outputs, name=name)
