#! -*- coding: utf-8 -*-
""" Pytorch ColBERT Common Modules
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from sim.pytorch.modeling_albert import ALBERT
from sim.pytorch.modeling_bert import BertModel
from sim.pytorch.modeling_nezha import NEZHA
from sim.tools import BertConfig
from sim.tools.tokenizer import BertTokenizer


class ColBERT(nn.Module):
    """ColBERT
    """

    def __init__(self,
                 config: BertConfig,
                 batch_size: int,
                 bert_model_type: str = "bert",
                 feature_dim: int = 128,
                 mask_punctuation: bool = False,
                 tokenizer: BertTokenizer = None,
                 similarity_metric: str = "cosine"):
        """
        :param config: BertConfig实例
        :param batch_size: batch size
        :param bert_model_type: bert模型
        :param feature_dim: feature size
        :param mask_punctuation: 是否mask标点符号
        :param tokenizer: 编码器，如果mask_punctuation为True，必传
        :param similarity_metric: 相似度计算方式
        """
        super(ColBERT, self).__init__()
        self.mask_punctuation = mask_punctuation
        self.similarity_metric = similarity_metric

        if bert_model_type == "bert":
            self.bert_model = BertModel(config=config, batch_size=batch_size)
        elif bert_model_type == "albert":
            self.bert_model = ALBERT(config=config, batch_size=batch_size)
        elif bert_model_type == "nezha":
            self.bert_model = NEZHA(config=config, batch_size=batch_size)
        else:
            raise ValueError("`model_type` must in bert/albert/nezha")

        self.filter_dense = nn.Linear(in_features=config.hidden_size, out_features=feature_dim, bias=False)
        if mask_punctuation:
            skip_list = {w: True for symbol in string.punctuation for w in [symbol, tokenizer.encode(symbol)[0][1]]}

    def forward(self, query_input_ids, query_token_type_ids, doc_input_ids, doc_token_type_ids):
        query_embedding = self.bert_model(query_input_ids, query_token_type_ids)
        query_outputs = self.filter_dense(query_embedding)
        query_outputs = F.normalize(input=query_outputs, p=2, dim=2)

        doc_embedding = self.bert_model(doc_input_ids, doc_token_type_ids)
        doc_outputs = self.filter_dense(doc_embedding)
        if self.mask_punctuation:
            mask = [[(token not in self.skip_list) and (token != 0) for token in doc] for doc in doc_input_ids.tolist()]
            mask = torch.tensor(data=mask, dtype=torch.float32).unsqueeze(dim=-1)
            doc_outputs = doc_outputs * mask
        doc_outputs = F.normalize(input=doc_outputs, p=2, dim=2)

        if self.similarity_metric == "cosine":
            outputs = (query_outputs @ doc_outputs.permute(0, 2, 1)).max(2).values.sum(1)
        elif self.similarity_metric == "l2":
            outputs = (query_outputs.unsqueeze(2) - doc_outputs.unsqueeze(1)) ** 2
            outputs = (-1.0 * outputs.sum(-1)).max(-1).values.sum(-1)
        else:
            raise ValueError("`similarity_metric` must be cosine or l2")

        return outputs
