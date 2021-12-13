#! -*- coding: utf-8 -*-
""" Implementation of Bert
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from sim.bert_base import BertConfig
from typing import Any


class BertEmbedding(nn.Module):
    """Bert Embedding"""

    def __init__(self, config: BertConfig):
        super(BertEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(num_embeddings=config.max_position_embeddings,
                                                embedding_dim=config.hidden_size)
        self.token_type_embeddings = nn.Embedding(num_embeddings=config.type_vocab_size,
                                                  embedding_dim=config.hidden_size)
        self.diff_token_type_embeddings = nn.Embedding(num_embeddings=5, embedding_dim=config.hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # 这里把position ids写到缓存里
        self.register_buffer("position_ids", torch.arange(end=config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.segment_type = getattr(config, "segment_type", "absolute")

    def forward(self, input_ids: Any = None, token_type_ids: Any = None, position_ids: Any = None,
                inputs_embeds: Any = None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_len = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, 0:seq_len]

        if token_type_ids is None:
            token_type_ids = torch.zeros(size=input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if self.segment_type == "absolute":
            segment_token_type_ids = token_type_ids % 10
            diff_token_type_ids = token_type_ids // 10
            segment_token_type_embeddings = self.token_type_embeddings(segment_token_type_ids)
            diff_token_type_embeddings = self.diff_token_type_embeddings(diff_token_type_ids)
            # token_type_embeddings = segment_token_type_embeddings + diff_token_type_embeddings
            token_type_embeddings = segment_token_type_embeddings
        else:
            token_type_embeddings = self.token_type_embeddings(torch.zeros_like(input=token_type_ids))

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
