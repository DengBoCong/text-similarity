#! -*- coding: utf-8 -*-
""" Tensorflow AlBert Common Modules
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
from sim.tensorflow.modeling_bert import bert_embedding
from sim.tensorflow.modeling_bert import BertLayer
from sim.tensorflow.layers import BertOutput
from sim.tensorflow.layers import Embedding
from sim.tools import BertConfig
from typing import Any


def albert(config: BertConfig,
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

    bert_layer = BertLayer(config=config, batch_size=batch_size, name="bert-layer")
    for index in range(config.num_hidden_layers):
        outputs = bert_layer([outputs, input_mask])

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
