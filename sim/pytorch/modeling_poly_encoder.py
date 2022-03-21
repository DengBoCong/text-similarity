#! -*- coding: utf-8 -*-
""" Pytorch Poly-Encoder Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch as torch
import torch.nn as nn
from sim.pytorch.common import dot_product_attention
from sim.pytorch.common import get_activation
from sim.pytorch.common import truncated_normal_
from sim.pytorch.modeling_albert import ALBERT
from sim.pytorch.modeling_bert import BertModel
from sim.pytorch.modeling_nezha import NEZHA
from sim.tools import BertConfig


class PolyEncoder(nn.Module):
    """ Poly Encoder
    """

    def __init__(self,
                 config: BertConfig,
                 batch_size: int,
                 bert_model_type: str = "bert",
                 poly_type: str = "learnt",
                 candi_agg_type: str = "cls",
                 poly_m: int = 16,
                 has_labels: bool = True):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param bert_model_type: bert模型
        :param poly_type: m获取形式，learnt, first, last
        :param candi_agg_type: candidate表示类型，cls, avg
        :param poly_m: 控制阈值，m个global feature
        :param has_labels: 自监督/无监督
        """
        super(PolyEncoder, self).__init__()
        self.batch_size = batch_size
        self.embeddings_size = config.hidden_size
        self.poly_type = poly_type
        self.poly_m = poly_m
        self.candi_agg_type = candi_agg_type
        self.dropout_rate = config.hidden_dropout_prob
        self.has_labels = has_labels

        if poly_type == "learnt":
            self.poly_embeddings = nn.Embedding(num_embeddings=poly_m + 1, embedding_dim=self.embeddings_size)

        if bert_model_type == "bert":
            self.bert_model = BertModel(config=config, batch_size=batch_size)
        elif bert_model_type == "albert":
            self.bert_model = ALBERT(config=config, batch_size=batch_size)
        elif bert_model_type == "nezha":
            self.bert_model = NEZHA(config=config, batch_size=batch_size)
        else:
            raise ValueError("`model_type` must in bert/albert/nezha")

        if has_labels:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.dense = nn.Linear(in_features=self.embeddings_size, out_features=2)
            truncated_normal_(stddev=config.initializer_range)(self.dense.weight)

    def forward(self, context_input_ids, context_token_type_ids, candidate_input_ids, candidate_token_type_ids):
        context_embedding = self.bert_model(context_input_ids, context_token_type_ids)
        candidate_embedding = self.bert_model(candidate_input_ids, candidate_token_type_ids)

        if self.poly_type == "learnt":
            context_poly_code_ids = torch.arange(start=1, end=self.poly_m + 1, step=1)
            context_poly_code_ids = torch.unsqueeze(input=context_poly_code_ids, dim=0)
            context_poly_code_ids = context_poly_code_ids.expand(self.batch_size, self.poly_m)
            context_poly_codes = self.poly_embeddings(context_poly_code_ids)
            context_vec, _ = dot_product_attention(
                query=context_poly_codes, key=context_embedding, value=context_embedding,
                depth=self.embeddings_size, dropout=self.dropout_rate
            )
        elif self.poly_type == "first":
            context_vec = context_embedding[:, :self.poly_m]
        elif self.poly_type == "last":
            context_vec = context_embedding[:, -self.poly_m:]
        else:
            raise ValueError("`poly_type` must in [learnt, first, last]")

        if self.candi_agg_type == "cls":
            candidate_vec = candidate_embedding[:, 0]
        elif self.candi_agg_type == "avg":
            candidate_vec = torch.sum(input=candidate_embedding, dim=1)
        else:
            raise ValueError("`candi_agg_type` must in [cls, avg]")

        final_vec, _ = dot_product_attention(query=candidate_vec, key=context_vec, value=context_vec,
                                             depth=self.embeddings_size, dropout=self.dropout_rate)
        outputs = torch.mean(input=final_vec * candidate_vec, dim=1)

        if self.has_labels:
            # 这里做二分类
            outputs = self.dropout(outputs)
            outputs = self.dense(outputs)
            outputs = get_activation("softmax")(input=outputs, dim=-1)

        return outputs
