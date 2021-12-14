#! -*- coding: utf-8 -*-
""" Implementation of Bert
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
from sim.bert_base import BertConfig
from sim.bert_base.torch.activations import ACT2FN
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

    def forward(self,
                input_ids: Any = None,
                token_type_ids: Any = None,
                position_ids: Any = None,
                inputs_embeds: Any = None):
        """
        :param input_ids: int32, [batch_size, seq_length]
        :param token_type_ids: token type embedding
        :param position_ids: position ids
        :param inputs_embeds: [batch_size, seq_length, emb_dim]
        """
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


class BertSelfAttention(nn.Module):
    """Bert Self-Attention"""

    def __init__(self, config: BertConfig):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.key = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.value = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_prob_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(num_embeddings=2 * config.max_position_embeddings - 1,
                                                   embedding_dim=self.attention_head_size)
        self.segment_type = getattr(config, "segment_type", "absolute")
        if self.segment_type == "relative":
            self.register_parameter("segment_embedding", param=nn.Parameter(torch.zeros()))

    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                output_attentions: bool = False):
        """
        :param hidden_states: float, [batch_size, seq_length, embedding_size]
        :param attention_mask: int, [batch_size, from_seq_length, to_seq_length]
        :param token_type_ids: int, [batch_size, seq_length]
        :param output_attentions: 是否输出attention
        """
        query_layer = self.transpose_for_scores(x=self.query(hidden_states))
        key_layer = self.transpose_for_scores(x=self.key(hidden_states))
        value_layer = self.transpose_for_scores(x=self.value(hidden_states))

        attention_scores = torch.matmul(input=query_layer, other=key_layer.transpose(-1, -2))
        if self.segment_type == "relative":
            segment_tensor = self.segment_embedding
            batch_size, seq_len = token_type_ids.size()
            token_type_ids_l = token_type_ids.view(batch_size, 1, seq_len)
            token_type_ids_r = token_type_ids.view(batch_size, seq_len, 1)
            token_type_distance = torch.abs(input=token_type_ids_l - token_type_ids_r)  # (b, l, r)
            token_type_distance = token_type_distance.unsqueeze(1)
            # (b,h,l,2)
            segment_embedding = torch.matmul(input=query_layer, other=segment_tensor.T)
            relative_segment_scores = torch.where(condition=token_type_distance > 0,
                                                  self=segment_embedding[..., 1:], other=segment_embedding[..., 0:1])
            attention_scores = attention_scores + relative_segment_scores
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_len = hidden_states.size()[1]
            position_ids_l = torch.arange(end=seq_len, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(end=seq_len, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_prob = nn.Softmax(dim=-1)(attention_scores)
        attention_prob = self.dropout(attention_prob)

        context_layer = torch.matmul(input=attention_prob, other=value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_prob) if output_attentions else (context_layer,)

        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        """
        :param hidden_states: float, [batch_size, seq_length, hidden_size]
        :param input_tensor: float, [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config=config)
        self.output = BertSelfOutput(config=config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                output_attentions: bool = False):
        """
        :param hidden_states: float, [batch_size, seq_length, embedding_size]
        :param attention_mask: int, [batch_size, from_seq_length, to_seq_length]
        :param token_type_ids: int, [batch_size, seq_length]
        :param output_attentions: 是否输出attention
        """
        self_outputs = self.self(hidden_states, attention_mask, token_type_ids, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        """
        :param hidden_states: float, [batch_size, seq_length, hidden_size]
        :param input_tensor: float, [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Bert Layer"""

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.seq_len_dim = 1
        self.attention = BertAttention(config=config)
        self.intermediate = BertIntermediate(config=config)
        self.output = BertOutput(config=config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                output_attentions: bool = False):
        """
        :param hidden_states: float, [batch_size, seq_length, embedding_size]
        :param attention_mask: int, [batch_size, from_seq_length, to_seq_length]
        :param token_type_ids: int, [batch_size, seq_length]
        :param output_attentions: 是否输出attention
        """
        self_attention_outputs = self.attention(hidden_states, attention_mask, token_type_ids, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """Bert Encoder"""

    def __init__(self, config: BertConfig):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config=config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                output_attentions: bool = False):
