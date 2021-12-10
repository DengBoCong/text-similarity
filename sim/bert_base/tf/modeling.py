#! -*- coding: utf-8 -*-
""" Implementation of Bert
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from sim.bert_base import BertConfig
from typing import NoReturn


class BertModel(object):
    """Bert Model"""

    def __init__(self,
                 config: BertConfig,
                 is_training: bool,
                 input_ids: tf.Tensor,
                 input_mask: tf.Tensor = None,
                 token_type_ids: tf.Tensor = None,
                 use_one_hot_embeddings: bool = False,
                 scope: str = "bert") -> NoReturn:
        """构建BertModel
        :param config: BertConfig实例
        :param is_training: train/eval
        :param input_ids: int32, [batch_size, seq_length]
        :param input_mask: int32, [batch_size, seq_length]
        :param token_type_ids: int32, [batch_size, seq_length]
        :param use_one_hot_embeddings: 是否使用one-hot embedding
        :param scope: 变量作用域
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_prob_dropout_prob = 0.0

        input_shape =

def get_shape_list
