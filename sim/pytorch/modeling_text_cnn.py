#! -*- coding: utf-8 -*-
""" Implementation of Text CNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from sim.pytorch.common import get_activation


class TextCNN(nn.Module):
    """Text CNN
    """

    def __init__(self,
                 seq_len: int,
                 embedding_size: int,
                 units: int,
                 filter_num: int,
                 kernel_sizes: list,
                 activations: list,
                 padding: str = "valid",
                 dropout: float = 0.1,
                 act: str = "tanh"):
        """
        :param seq_len: 序列长度
        :param embedding_size: 特征为大小
        :param units: 全连接层hidden size
        :param filter_num: 滤波器个数
        :param kernel_sizes: 卷积核大小，可以是多个
        :param activations: 激活函数列表，个数同kernel_sizes
        :param padding: 填充类型
        :param dropout: 采样率
        :param act: 全连接层激活函数
        """
        super(TextCNN, self).__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.units = units
        self.filter_num = filter_num
        self.kernel_sizes = kernel_sizes
        self.activations = activations
        self.padding = padding
        self.dropout = dropout
        self.act = act

        for index, kernel_size in enumerate(self.kernel_sizes):
            setattr(self, f"conv2d_{index}", nn.Conv2d(in_channels=1, out_channels=self.filter_num,
                                                       kernel_size=(kernel_size, self.embedding_size),
                                                       stride=(1, 1), padding=self.padding))
            setattr(self, f"pooled_{index}", nn.MaxPool2d(kernel_size=(seq_len - kernel_size + 1, 1), stride=(1, 1)))

        self.dropout = nn.Dropout(p=self.dropout)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=len(self.kernel_sizes) * self.filter_num, out_features=self.units)

    def forward(self, inputs):
        reshape_inputs = inputs.unsqueeze(dim=-1).permute(dims=(0, 3, 1, 2))
        conv_pools = []
        for index, activation in enumerate(self.activations):
            conv = getattr(self, f"conv2d_{index}")(reshape_inputs)
            conv_act = get_activation(activation)(conv)
            pooled = getattr(self, f"pooled_{index}")(conv_act)
            conv_pools.append(pooled)

        outputs = torch.concat(conv_pools, dim=-1)
        outputs = self.dropout(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        outputs = get_activation(self.act)(outputs)

        return outputs
