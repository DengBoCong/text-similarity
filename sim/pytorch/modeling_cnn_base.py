#! -*- coding: utf-8 -*-
""" Implementation of CNN Base
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
