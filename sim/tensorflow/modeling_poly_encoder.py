#! -*- coding: utf-8 -*-
""" Tensorflow Poly-Encoder Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras
from sim.tensorflow.common import dot_product_attention
from typing import Any


def poly_encoder(context_bert_model: Any,
                 candidate_bert_model: Any,
                 batch_size: int,
                 embeddings_size: int,
                 poly_type: str = "learnt",
                 candi_agg_type: str = "cls",
                 poly_m: int = 16,
                 dropout: float = 0.1,
                 initializer_range: float = 0.02,
                 has_labels: bool = True):
    """Poly Encoder
    :param context_bert_model: 预加载用于context编码句子的bert类
    :param candidate_bert_model: 预加载用于candidate编码句子的bert类
    :param batch_size: batch size
    :param embeddings_size: feature size
    :param poly_type: m获取形式，learnt, first, last
    :param candi_agg_type: candidate表示类型，cls, avg
    :param poly_m: 控制阈值，m个global feature
    :param dropout: 采样率
    :param initializer_range: 截断分布范围
    :param has_labels: 自监督/无监督
    """
    context_input_ids = keras.Input(shape=(None,))
    context_token_type_ids = keras.Input(shape=(None,))
    context_embedding = context_bert_model([context_input_ids, context_token_type_ids])

    candidate_input_ids = keras.Input(shape=(None,))
    candidate_token_type_ids = keras.Input(shape=(None,))
    candidate_embedding = candidate_bert_model([candidate_input_ids, candidate_token_type_ids])

    if poly_type == "learnt":
        context_poly_code_ids = keras.backend.arange(start=1, stop=poly_m + 1)
        context_poly_code_ids = tf.expand_dims(input=context_poly_code_ids, axis=0)
        context_poly_code_ids = tf.repeat(input=context_poly_code_ids, repeats=batch_size, axis=0)
        context_poly_codes = keras.layers.Embedding(input_dim=poly_m + 1,
                                                    output_dim=embeddings_size)(context_poly_code_ids)
        context_vec, _ = dot_product_attention(query=context_poly_codes, key=context_embedding,
                                               value=context_embedding, depth=embeddings_size, dropout=dropout)
    elif poly_type == "first":
        context_vec = context_embedding[:, :poly_m]
    elif poly_type == "last":
        context_vec = context_embedding[:, -poly_m:]
    else:
        raise ValueError("`poly_type` must in [learnt, first, last]")

    if candi_agg_type == "cls":
        candidate_vec = candidate_embedding[:, 0]
    elif candi_agg_type == "avg":
        candidate_vec = tf.reduce_mean(input_tensor=candidate_embedding, axis=1)
    else:
        raise ValueError("`candi_agg_type` must in [cls, avg]")

    final_vec, _ = dot_product_attention(query=candidate_vec, key=context_vec, value=context_vec,
                                         depth=embeddings_size, dropout=dropout)

    outputs = tf.reduce_mean(input_tensor=final_vec * candidate_vec, axis=1)

    if has_labels:
        # 这里做二分类
        outputs = keras.layers.Dropout(rate=dropout)(outputs)
        outputs = keras.layers.Dense(
            units=2, activation="softmax",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range)
        )(outputs)

    return keras.Model(inputs=[context_input_ids, context_token_type_ids, candidate_input_ids,
                               candidate_token_type_ids], outputs=outputs, name="poly-encoder")
