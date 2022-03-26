#! -*- coding: utf-8 -*-
""" Tensorflow ColBERT Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import tensorflow as tf
import tensorflow.keras as keras
from sim.tools.tokenizer import BertTokenizer
from typing import Any


def colbert(encoder_model: Any,
            feature_dim: int = 128,
            mask_punctuation: bool = False,
            tokenizer: BertTokenizer = None,
            similarity_metric: str = "cosine") -> keras.Model:
    """ColBERT
    :param encoder_model: 对query和doc共用encoder
    :param feature_dim: feature size
    :param mask_punctuation: 是否mask标点符号
    :param tokenizer: 编码器，如果mask_punctuation为True，必传
    :param similarity_metric: 相似度计算方式
    """
    query_input_ids = keras.Input(shape=(None,), dtype=tf.int32)
    query_token_type_ids = keras.Input(shape=(None,), dtype=tf.int32)

    doc_input_ids = keras.Input(shape=(None,))
    doc_token_type_ids = keras.Input(shape=(None,))

    if mask_punctuation:
        skip_list = {w: True for symbol in string.punctuation for w in [symbol, tokenizer.encode(symbol)[0][1]]}
        mask = [[(token not in skip_list) and (token != 0) for token in doc] for doc in doc_input_ids.numpy().tolist()]
        mask = tf.expand_dims(input=tf.convert_to_tensor(value=mask, dtype=tf.float32), axis=-1)

    filter_dense = keras.layers.Dense(units=feature_dim, use_bias=False)

    query_embedding = encoder_model([query_input_ids, query_token_type_ids])
    query_outputs = filter_dense(query_embedding)
    query_outputs = tf.divide(x=query_outputs, y=tf.norm(tensor=query_outputs, ord=2, axis=2, keepdims=True))

    doc_embedding = encoder_model([doc_input_ids, doc_token_type_ids])
    doc_outputs = filter_dense(doc_embedding)
    if mask_punctuation:
        doc_outputs = doc_outputs * mask
    doc_outputs = tf.divide(x=doc_outputs, y=tf.norm(tensor=doc_outputs, ord=2, axis=2, keepdims=True))

    if similarity_metric == "cosine":
        multi = tf.matmul(a=query_outputs, b=doc_outputs, transpose_b=True)
        outputs = tf.reduce_sum(input_tensor=tf.reduce_max(input_tensor=multi, axis=-1), axis=-1)

    elif similarity_metric == "l2":
        query_outputs = tf.expand_dims(input=query_outputs, axis=2)
        doc_outputs = tf.expand_dims(input=doc_outputs, axis=1)
        outputs = tf.reduce_sum(input_tensor=(query_outputs - doc_outputs) ** 2, axis=-1)
        outputs = tf.reduce_max(input_tensor=(-1. * outputs), axis=-1)
        outputs = tf.reduce_sum(input_tensor=outputs, axis=-1)
    else:
        raise ValueError("`similarity_metric` must be cosine or l2")

    return keras.Model(inputs=[query_input_ids, query_token_type_ids, doc_input_ids, doc_token_type_ids],
                       outputs=outputs, name="colbert")
