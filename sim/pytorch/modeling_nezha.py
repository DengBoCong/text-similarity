#! -*- coding: utf-8 -*-
""" Pytorch NeZha Common Modules
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
from sim.pytorch.common import sinusoidal_init_
from sim.pytorch.common import truncated_normal_
from sim.pytorch.layers import BertOutput
from sim.pytorch.layers import BertSelfAttention
from sim.pytorch.layers import Embedding
from sim.pytorch.layers import FeedForward
from sim.pytorch.layers import RelativePositionEmbedding
from sim.tools import BertConfig
from typing import Any


class BertEmbeddings(nn.Module):
    """Bert Embedding
    """

    def __init__(self,
                 hidden_size: int,
                 embedding_size: int,
                 hidden_dropout_prob: float = None,
                 shared_segment_embeddings: bool = False,
                 type_vocab_size: int = None,
                 layer_norm_eps: float = None,
                 initializer: Any = truncated_normal_()):
        """NEZHA Embedding
        :param hidden_size: 编码维度
        :param embedding_size: 词嵌入大小
        :param hidden_dropout_prob: Dropout比例
        :param shared_segment_embeddings: 若True，则segment跟token共用embedding
        :param type_vocab_size: segment总数目
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param initializer: Embedding的初始化器
        """
        super(BertEmbeddings, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.shared_segment_embeddings = shared_segment_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer = initializer

        if self.type_vocab_size > 0 and not self.shared_segment_embeddings:
            self.segment_embeddings = nn.Embedding(
                num_embeddings=self.type_vocab_size,
                embedding_dim=self.embedding_size
            )
            self.initializer(self.segment_embeddings.weight)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)

        if self.embedding_size != self.hidden_size:
            self.outputs_dense = nn.Linear(in_features=self.embedding_size, out_features=self.hidden_size)

    def forward(self, input_ids, segment_ids, token_embeddings):
        outputs = token_embeddings(input_ids)

        if self.type_vocab_size > 0:
            if self.shared_segment_embeddings:
                segment_outputs = token_embeddings(segment_ids)
            else:
                segment_outputs = self.segment_embeddings(segment_ids)

            outputs = outputs + segment_outputs

        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)

        if self.embedding_size != self.hidden_size:
            outputs = self.outputs_dense(outputs)

        return outputs


class BertLayer(nn.Module):
    """Bert Block
    """

    def __init__(self, config: BertConfig, batch_size: int, initializer: Any = None):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param initializer: 初始化器
        """
        super(BertLayer, self).__init__()
        self.bert_config = config
        self.batch_size = batch_size
        self.initializer = initializer if initializer else truncated_normal_(stddev=config.initializer_range)
        self.key_size = self.bert_config.attention_key_size

        emb_input_dim = 2 * 64 + 1
        emb_output_dim = self.key_size if self.key_size is not None else self.bert_config.attention_head_size
        self.embeddings_initializer = sinusoidal_init_(position=emb_input_dim, depth=emb_output_dim)

        self.position_embeddings = RelativePositionEmbedding(
            input_dim=emb_input_dim,
            output_dim=emb_output_dim,
            initializer=self.embeddings_initializer,
            requires_grad=False
        )

        self.bert_self_attention = BertSelfAttention(
            num_heads=self.bert_config.num_attention_heads,
            head_size=self.bert_config.attention_head_size,
            batch_size=self.batch_size,
            attention_dropout=self.bert_config.attention_probs_dropout_prob,
            key_size=self.bert_config.attention_key_size,
            hidden_size=self.bert_config.hidden_size,
            initializer=self.initializer,
            pos_type="typical_relation"
        )
        self.attn_dropout = nn.Dropout(p=self.bert_config.hidden_dropout_prob)
        self.attn_norm = nn.LayerNorm(normalized_shape=self.bert_config.hidden_size,
                                      eps=self.bert_config.layer_norm_eps)
        self.feedforward = FeedForward(
            in_features=self.bert_config.hidden_size,
            mid_features=self.bert_config.intermediate_size,
            out_features=self.bert_config.hidden_size,
            activation=self.bert_config.hidden_act,
            initializer=self.initializer
        )

        self.feedforward_dropout = nn.Dropout(p=self.bert_config.hidden_dropout_prob)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=self.bert_config.hidden_size,
                                             eps=self.bert_config.layer_norm_eps)

    def forward(self, inputs, mask):
        pos_ids = self.position_embeddings(inputs, inputs)
        attn_outputs, attn_weights = self.bert_self_attention([inputs, inputs, inputs, pos_ids, mask])
        attn_outputs = self.attn_dropout(attn_outputs)
        attn_outputs = attn_outputs + inputs
        attn_outputs = self.attn_norm(attn_outputs)

        outputs = self.feedforward(attn_outputs)
        outputs = self.feedforward_dropout(outputs)
        outputs = outputs + attn_outputs
        outputs = self.feedforward_norm(outputs)

        return outputs


class NEZHA(nn.Module):
    """NEZHA Model
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
        super(NEZHA, self).__init__()
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
            type_vocab_size=self.config.type_vocab_size,
            layer_norm_eps=self.config.layer_norm_eps,
            initializer=self.initializer
        )

        for index in range(self.config.num_hidden_layers):
            setattr(self, f"bert_layer_{index}", BertLayer(
                config=self.config, batch_size=self.batch_size, initializer=self.initializer
            ))

        if self.add_pooling_layer:
            argument = {}
            if with_mlm:
                argument["embedding_size"] = self.config.embedding_size
                argument["hidden_act"] = self.config.hidden_act
                argument["layer_norm_eps"] = self.config.layer_norm_eps
                argument["mlm_decoder"] = self.token_embeddings
                argument["mlm_decoder_arg"] = {"mode": "dense"}
                argument["vocab_size"] = self.config.vocab_size

            self.bert_output = BertOutput(with_pool, with_nsp, with_mlm,
                                          self.initializer, self.config.hidden_size, **argument)

    def forward(self, input_ids, token_type_ids):
        input_mask = torch.eq(input=input_ids, other=0).float()[:, None, None, :]

        outputs = self.bert_embeddings(input_ids, token_type_ids, self.token_embeddings)
        for index in range(self.config.num_hidden_layers):
            outputs = getattr(self, f"bert_layer_{index}")(outputs, input_mask)

        if self.add_pooling_layer:
            outputs = self.bert_output(outputs)

        return outputs
