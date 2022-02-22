#! -*- coding: utf-8 -*-
""" Pytorch AlBert Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import torch
import torch.nn as nn
from sim.pytorch.common import truncated_normal_
from sim.pytorch.layers import BertOutput
from sim.pytorch.layers import Embedding
from sim.pytorch.modeling_bert import BertEmbeddings
from sim.pytorch.modeling_bert import BertLayer
from sim.tools import BertConfig
from typing import Any


class ALBERT(nn.Module):
    """ALBERT Model
    """

    def __init__(self,
                 config: BertConfig,
                 batch_size: int,
                 position_merge_mode: str = "add",
                 is_training: bool = True,
                 add_pooling_layer: bool = True,
                 with_pool: Any = False,
                 with_nsp: Any = False,
                 with_mlm: Any = False):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param position_merge_mode: 输入和position合并的方式
        :param is_training: train/eval
        :param add_pooling_layer: 处理输出，后面三个参数用于此
        :param with_pool: 是否包含Pool部分, 必传hidden_size
        :param with_nsp: 是否包含NSP部分
        :param with_mlm: 是否包含MLM部分, 必传embedding_size, hidden_act, layer_norm_eps, mlm_decoder
        """
        super(ALBERT, self).__init__()
        self.config = copy.deepcopy(config)
        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_prob_dropout_prob = 0.0

        self.batch_size = batch_size
        self.position_merge_mode = position_merge_mode
        self.is_training = is_training
        self.add_pooling_layer = add_pooling_layer
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

        self.initializer = truncated_normal_(mean=0., stddev=self.config.initializer_range)
        self.token_embeddings = Embedding(num_embeddings=self.config.vocab_size,
                                          embedding_dim=self.config.embedding_size, padding_idx=0)
        self.initializer(self.token_embeddings.weight)

        self.bert_embeddings = BertEmbeddings(
            hidden_size=self.config.hidden_size,
            embedding_size=self.config.embedding_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            shared_segment_embeddings=self.config.shared_segment_embeddings,
            max_position=self.config.max_position,
            position_merge_mode=self.position_merge_mode,
            hierarchical_position=self.config.hierarchical_position,
            type_vocab_size=self.config.type_vocab_size,
            layer_norm_eps=self.config.layer_norm_eps,
            initializer=self.initializer
        )

        self.bert_layer = BertLayer(config=self.config, batch_size=self.batch_size, initializer=self.initializer)

        if self.add_pooling_layer:
            argument = {}
            if with_mlm:
                argument["embedding_size"] = self.config.embedding_size
                argument["hidden_act"] = self.config.hidden_act
                argument["layer_norm_eps"] = self.config.layer_norm_eps
                argument["mlm_decoder"] = nn.Linear(in_features=self.config.embedding_size,
                                                    out_features=self.config.vocab_size)
                argument["vocab_size"] = self.config.vocab_size

            self.bert_output = BertOutput(with_pool, with_nsp, with_mlm,
                                          self.initializer, self.config.hidden_size, **argument)

    def forward(self, input_ids, token_type_ids):
        input_mask = torch.eq(input=input_ids, other=0).float()[:, None, None, :]

        outputs = self.bert_embeddings(input_ids, token_type_ids, self.token_embeddings)
        for index in range(self.config.num_hidden_layers):
            outputs = self.bert_layer(outputs, input_mask)

        if self.add_pooling_layer:
            outputs = self.bert_output(outputs)

        return outputs
