#! -*- coding: utf-8 -*-
""" Implementation of Siamese RNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from typing import Any
from typing import NoReturn


class SiameseRnnWithEmbedding(nn.Module):
    """ Siamese LSTM with Embedding """

    def __init__(self, emb_dim: int, vocab_size: int, units: int, dropout: float,
                 num_layers: int, rnn: str, share: bool = True, if_bi: bool = True) -> NoReturn:
        """
        :param emb_dim: embedding dim
        :param vocab_size: 词表大小，例如为token最大整数index + 1.
        :param units: 输出空间的维度
        :param dropout: 采样率
        :param num_layers: RNN层数
        :param rnn: RNN的实现类型
        :param share: 是否共享权重
        :param if_bi: 是否双向
        :return: Model
        """
        super(SiameseRnnWithEmbedding, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.embedding2 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

        if rnn not in ["lstm", "gru"]:
            raise ValueError("{} is unknown type".format(rnn))

        if rnn == "lstm":
            self.rnn1 = nn.LSTM(input_size=emb_dim, hidden_size=units,
                                num_layers=num_layers, bidirectional=if_bi)
            self.rnn2 = nn.LSTM(input_size=emb_dim, hidden_size=units,
                                num_layers=num_layers, bidirectional=if_bi)
        elif rnn == "gru":
            self.rnn1 = nn.GRU(input_size=emb_dim, hidden_size=units,
                               num_layers=num_layers, bidirectional=if_bi)
            self.rnn2 = nn.GRU(input_size=emb_dim, hidden_size=units,
                               num_layers=num_layers, bidirectional=if_bi)

        self.if_bi = if_bi
        self.share = share
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs1: Any, inputs2: Any) -> tuple:
        """
        :param inputs1:
        :param inputs2:
        """
        embedding1 = self.embedding1(inputs1)
        embedding2 = self.embedding2(inputs2)

        dropout1 = self.dropout(embedding1)
        dropout2 = self.dropout(embedding2)

        if self.share:
            outputs1 = self.rnn1(dropout1)
            outputs2 = self.rnn1(dropout2)
        else:
            outputs1 = self.rnn1(dropout1)
            outputs2 = self.rnn2(dropout2)

        # 这里如果使用了双向，需要将两个方向的特征层合并起来，维度将会是units * 2
        if self.if_bi:
            state1 = torch.cat((outputs1[1][0][-2, :, :], outputs1[1][0][-1, :, :]), dim=-1)
            state2 = torch.cat((outputs2[1][0][-2, :, :], outputs2[1][0][-1, :, :]), dim=-1)

            return state1, state2

        return outputs1[1][0][-1:, :, :], outputs2[1][0][-1, :, :]
