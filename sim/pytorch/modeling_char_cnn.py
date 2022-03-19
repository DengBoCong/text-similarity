#! -*- coding: utf-8 -*-
""" Implementation of Char CNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from sim.pytorch.common import get_activation


class CharCNN(nn.Module):
    """Char CNN
    """

    def __init__(self,
                 seq_len: int,
                 embeddings_size: int,
                 word_max_len: int,
                 char_cnn_layers: list,
                 highway_layers: int = 2,
                 num_rnn_layers: int = 2,
                 rnn_units: int = 650,
                 dropout: float = 0.5,
                 label_num: int = 2,
                 label_act: str = "softmax"):
        """
        :param seq_len: 序列长度
        :param embeddings_size: 特征大小
        :param word_max_len: 单个token最大长度
        :param char_cnn_layers: 多层卷积列表，(filter_num, kernel_size_1)
                [[50, 1], [100, 2], [150, 3], [200, 4], [200, 5], [200, 6], [200, 7]]
        :param highway_layers: 使用highway层数
        :param num_rnn_layers: rnn层数
        :param rnn_units: rnn隐层大小
        :param dropout: 采样率
        :param label_num: 输出类别数
        :param label_act: 输出激活函数
        """
        super(CharCNN, self).__init__()
        self.seq_len = seq_len
        self.embeddings_size = embeddings_size
        self.word_max_len = word_max_len
        self.char_cnn_layers = char_cnn_layers
        self.highway_layers = highway_layers
        self.num_run_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.label_num = label_num
        self.label_act = label_act

        for index, char_cnn_size in enumerate(self.char_cnn_layers):
            setattr(self, f"conv_{index}", nn.Conv2d(
                in_channels=self.embeddings_size, out_channels=char_cnn_size[0], kernel_size=(1, char_cnn_size[1])
            ))

            setattr(self, f"pool_{index}", nn.MaxPool2d(
                kernel_size=(1, self.word_max_len - char_cnn_size[1] + 1)
            ))

        self.sum_filter_num = sum(np.array([ccl[0] for ccl in char_cnn_layers]))
        self.batch_norm = nn.BatchNorm1d(num_features=self.sum_filter_num)

        for highway_layer in range(highway_layers):
            setattr(self, f"highway_{highway_layer}", Highway(feature_dim=self.sum_filter_num))

        for index in range(num_rnn_layers):
            setattr(self, f"bi_lstm_{index}", nn.LSTM(
                input_size=self.sum_filter_num, hidden_size=rnn_units,
                bias=True, batch_first=True, bidirectional=True
            ))
            setattr(self, f"lstm_dropout_{index}", nn.Dropout(p=self.dropout))

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=self.seq_len * self.rnn_units * 2,
                               out_features=self.label_num)

    def __int__(self, inputs):
        embeddings = inputs.unsqueeze(dim=-1)
        concat_embeddings = torch.concat(tensors=[embeddings for i in range(self.word_max_len)], dim=-1)
        embeddings_outputs = torch.permute(input=concat_embeddings, dims=[0, 2, 1, 3])
        conv_out = []
        for index, char_cnn_size in enumerate(self.char_cnn_layers):
            conv = getattr(self, f"conv_{index}")(embeddings_outputs)
            conv = get_activation("tanh")(conv)
            pooled = getattr(self, f"pool_{index}")(conv)
            pooled = torch.permute(input=pooled, dims=[0, 2, 3, 1])
            conv_out.append(pooled)

        outputs = torch.concat(tensors=conv_out, dim=-1)
        outputs = torch.reshape(input=outputs, shape=(outputs.shape[0], self.seq_len,
                                                      outputs.shape[2] * self.sum_filter_num))
        outputs = self.batch_norm(outputs)

        for highway_layer in range(self.highway_layers):
            outputs = getattr(self, f"highway_{highway_layer}")(outputs)

        for index in range(self.num_rnn_layers):
            outputs = getattr(self, f"bi_lstm_{index}")(outputs)[0]
            outputs = getattr(self, f"lstm_dropout_{index}")(outputs)

        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        outputs = get_activation(self.label_act)(outputs)

        return outputs


class Highway(nn.Module):
    """Highway
    """

    def __init__(self, feature_dim: int, transform_gate_bias: int = -2):
        """
        :param feature_dim: 输入特征大小
        :param transform_gate_bias: 常数初始化器scalar
        """
        super(Highway, self).__init__()
        self.transform_gate_bias = transform_gate_bias

        self.gate_transform = nn.Linear(in_features=feature_dim, out_features=feature_dim)
        nn.init.constant_(tensor=self.gate_transform.weight, val=self.transform_gate_bias)
        self.block_state = nn.Linear(in_features=feature_dim, out_features=feature_dim)
        nn.init.zeros_(tensor=self.block_state.weight)

    def forward(self, inputs):
        gate_transform = self.gate_transform(inputs)
        gate_transform = get_activation("sigmoid")(gate_transform)
        gate_cross = 1. - gate_transform
        block_state = self.block_state(inputs)
        block_state = get_activation("gelu")(block_state)
        highway = gate_transform * block_state + gate_cross * inputs
        return highway
