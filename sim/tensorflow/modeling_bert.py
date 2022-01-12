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
from sim.tensorflow.layers import bert_self_attention
from sim.tensorflow.layers import feed_forward
from sim.tensorflow.layers import PositionEmbedding
from sim.tensorflow.layers import scaled_dot_product_attention
from typing import Any


def bert_embedding(vocab_size: int,
                   hidden_size: int,
                   embedding_size: int,
                   is_training: bool,
                   hidden_dropout_prob: float = None,
                   shared_segment_embeddings: bool = False,
                   max_position: int = 512,
                   position_merge_mode: str = "add",
                   hierarchical_position: Any = None,
                   segment_vocab_size: int = 2,
                   layer_norm_eps: float = 1e-12,
                   manual_seed: int = 1,
                   initializer: Any = None,
                   position_ids: Any = None,
                   name: str = "embedding") -> keras.Model:
    """Bert Embedding
    :param vocab_size: 词表大小
    :param hidden_size: 编码维度
    :param embedding_size: 词嵌入大小
    :param is_training: 是否处于训练模式
    :param hidden_dropout_prob: Dropout比例
    :param shared_segment_embeddings: 若True，则segment跟token共用embedding
    :param max_position: 绝对位置编码最大位置数
    :param position_merge_mode: 输入和position合并的方式
    :param hierarchical_position: 是否层次分解位置编码
    :param segment_vocab_size: segment总数目
    :param layer_norm_eps: layer norm 附加因子，避免除零
    :param manual_seed: 随机种子
    :param initializer: Embedding的初始化器
    :param position_ids: 位置编码ids
    :param name: 模型名
    """
    input_ids = keras.Input(shape=(None,))
    segment_ids = keras.Input(shape=(None,))

    # 默认使用截断正态分布初始化
    if not initializer:
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)

    word_embeddings = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        embeddings_initializer=initializer,
        mask_zero=True,
        name=f"{name}-token"
    )
    outputs = word_embeddings(input_ids)

    if segment_vocab_size > 0:
        if shared_segment_embeddings:
            segment_embeddings = word_embeddings(segment_ids)
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

    outputs = keras.layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}-norm")(outputs)
    outputs = keras.layers.Dropout(rate=hidden_dropout_prob,
                                   seed=manual_seed, name=f"{name}-dropout")(outputs, is_training)

    if embedding_size != hidden_size:
        outputs = keras.layers.Dense(
            units=hidden_size,
            kernel_initializer=initializer,
            name=f"{name}-mapping"
        )(outputs)

    return keras.Model(inputs=[input_ids, segment_ids], outputs=outputs, name=name)


def bert_layer(config: BertConfig,
               is_training: bool,
               seq_len: int,
               feature_dim: int,
               initializer: Any = None,
               manual_seed: int = 1,
               name: str = "bert") -> keras.Model:
    """Bert Layer
    :param config: BertConfig实例
    :param is_training: 是否处于训练模式
    :param seq_len: 序列长度
    :param feature_dim: 输入最后一个维度
    :param initializer: 初始化器
    :param manual_seed: 随机种子
    :param name: 模型名
    """
    hidden_states = tf.keras.Input(shape=(None, feature_dim))
    mask = tf.keras.Input(shape=(None, None, None))
    attn_name = f"{name}-multi-head-self-attention"
    feed_forward_name = f"{name}-feedforward"

    # 默认使用截断正态分布初始化
    if not initializer:
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)

    attn_outputs, attn_weights = bert_self_attention(
        num_heads=config.num_attention_heads,
        head_size=config.attention_head_size,
        attention_func=scaled_dot_product_attention,
        is_training=is_training,
        attention_dropout=config.attention_probs_dropout_prob,
        seq_len=seq_len,
        feature_dim=hidden_states.shape[-1],
        key_size=config.attention_key_size,
        hidden_size=config.hidden_size,
        initializer=initializer,
        manual_seed=manual_seed,
        name=attn_name
    )(hidden_states, hidden_states, hidden_states, mask)

    attn_outputs = keras.layers.Dropout(rate=config.hidden_dropout_prob, seed=manual_seed,
                                        name=f"{attn_name}-dropout")(attn_outputs, is_training)
    attn_outputs = keras.layers.Add(name=f"{attn_name}-add")([attn_outputs, hidden_states])
    attn_outputs = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps,
                                                   name=f"{attn_name}-norm")(attn_outputs)

    outputs = feed_forward(units=config.intermediate_size, activation=config.hidden_act,
                           kernel_initializer=initializer, name=feed_forward_name)(attn_outputs)
    outputs = keras.layers.Dropout(rate=config.hidden_dropout_prob, seed=manual_seed,
                                   name=f"{feed_forward_name}-dropout")(outputs, is_training)
    outputs = keras.layers.Add(name=f"{feed_forward_name}-add")([attn_outputs, outputs])
    outputs = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps,
                                              name=f"{feed_forward_name}-norm")(outputs)

    return keras.Model(inputs=[hidden_states, mask], outputs=outputs, name=name)


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
