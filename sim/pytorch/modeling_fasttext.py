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
import torch.nn.functional as F


class FastText(nn.Module):
    """ Fast Text Model
    """

    def __init__(self,
                 embedding_size: int,
                 seq_len: int,
                 hidden_size: int,
                 act: str = "tanh",
                 label_size: int = 2,
                 dropout: float = 0.1):
        """
        :param embedding_size: 特征维度大小
        :param seq_len: 序列长度
        :param hidden_size: 中间隐层大小
        :param act: 激活函数
        :param label_size: 类别数
        """
        super(FastText, self).__init__()
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.act = act
        self.label_size = label_size
        self.dropout = dropout

        self.max_pooling = nn.MaxPool1d(kernel_size=self.embedding_size)
        self.avg_pooling = nn.AvgPool1d(kernel_size=self.embedding_size)
        self.mid_dense = nn.Linear(in_features=self.seq_len * 2, out_features=self.hidden_size)
        self.mid_dropout = nn.Dropout(p=self.dropout)
        self.output_dense = nn.Linear(in_features=self.hidden_size, out_features=self.label_size)

    def forward(self, inputs):
        inputs_m = self.max_pooling(inputs).squeeze()
        inputs_a = self.avg_pooling(inputs).squeeze()

        outputs = torch.concat([inputs_m, inputs_a], dim=-1)
        outputs = self.mid_dense(outputs)
        outputs = F.tanh(outputs)
        outputs = self.mid_dropout(outputs)
        outputs = self.output_dense(outputs)
        outputs = F.softmax(outputs, dim=-1)

        return outputs
