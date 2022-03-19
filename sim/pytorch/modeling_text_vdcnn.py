#! -*- coding: utf-8 -*-
""" Implementation of Text VDCNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from sim.pytorch.layers import get_activation
from sim.pytorch.layers import SpatialDropout
from typing import Any


class KMaxPooling(nn.Module):
    """动态K-max pooling
     k的选择为 k = max(k, s * (L-1) / L)
     其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
    """

    def __init__(self, top_k: int = 8):
        super(KMaxPooling, self).__init__()
        self.top_k = top_k

    def forward(self, inputs):
        outputs = torch.topk(input=inputs, k=self.top_k, sorted=False).values
        return outputs


class ConvolutionalBlock(nn.Module):
    """Convolutional block
    """

    def __init__(self,
                 in_channels: int,
                 filter_num: int = 256,
                 kernel_size: int = 3,
                 strides: Any = 1,
                 padding: str = "same",
                 activation: Any = "linear"):
        """
        :param in_channels:
        :param filter_num: 滤波器大小
        :param kernel_size: 卷积核大小
        :param strides: 移动步幅
        :param padding: 填充类型
        :param activation: 激活函数
        """
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=filter_num,
                               kernel_size=kernel_size, padding=padding, stride=strides)
        self.batch_norm1 = nn.BatchNorm1d(num_features=filter_num)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=filter_num, out_channels=filter_num,
                               kernel_size=kernel_size, stride=strides, padding=padding)
        self.batch_norm2 = nn.BatchNorm1d(num_features=filter_num)
        self.relu2 = nn.ReLU()

        self.activation = activation

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = get_activation(self.activation)(outputs)
        outputs = self.batch_norm1(outputs)
        outputs = self.relu1(outputs)

        outputs = self.conv2(outputs)
        outputs = get_activation(self.activation)(outputs)
        outputs = self.batch_norm2(outputs)
        outputs = self.relu2(outputs)

        return outputs


class DownSampling(nn.Module):
    def __init__(self,
                 pool_type: str = "max",
                 pool_size: Any = 3,
                 strides: Any = 2,
                 top_k: int = None,
                 in_channels: int = None,
                 padding: str = "same"):
        """
        :param pool_type: "max", "k-max", "conv"
        :param pool_size: 池化窗口大小
        :param strides: 移动步幅
        :param top_k: top k, 如果是k-max，必传
        :param in_channels: 如果是conv，必传
        :param padding: 填充类型
        """
        super(DownSampling, self).__init__()
        if pool_type == "max":
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=strides, padding=padding)
        elif pool_type == "k-max":
            self.pool = KMaxPooling(top_k=top_k)
        elif pool_type == "conv":
            self.pool = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                  kernel_size=pool_size, stride=strides, padding=padding)
        else:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=strides, padding=padding)

    def forward(self, inputs):
        outputs = self.pool(inputs)
        return outputs


class ShortcutPool(nn.Module):
    """shortcut连接. 恒等映射, block+f(block)，加上 down sampling
    """

    def __init__(self,
                 in_channels: int,
                 filter_num: int = 256,
                 kernel_size: Any = 1,
                 strides: Any = 2,
                 padding: str = "same",
                 pool_type: str = "max",
                 shortcut: bool = True):
        """
        :param in_channels:
        :param filter_num: 滤波器大小
        :param kernel_size: 卷积核大小
        :param strides: 移动步幅
        :param padding: 填充类型
        :param pool_type: "max", "k-max", "conv"
        :param shortcut: 是否开启shortcut连接
        """
        super(ShortcutPool, self).__init__()
        self.shortcut = shortcut
        self.pool_type = pool_type

        if shortcut:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=filter_num,
                                  kernel_size=kernel_size, stride=strides, padding=padding)
            self.batch_norm = nn.BatchNorm1d(num_features=filter_num)
            self.down_sampling = DownSampling(self.pool_type)
        else:
            self.relu = nn.ReLU()
            self.down_sampling = DownSampling(self.pool_type)

        if pool_type is not None:
            self.conv1 = nn.Conv1d(in_channels=filter_num, out_channels=filter_num * 2,
                                   kernel_size=kernel_size, stride=strides, padding=padding)
            self.batch_norm1 = nn.BatchNorm1d(num_features=filter_num * 2)

    def forward(self, inputs, inputs_mid):
        if self.shortcut:
            conv = self.conv(inputs)
            conv = self.batch_norm(conv)
            outputs = self.down_sampling(inputs_mid)
            outputs = conv + outputs
        else:
            outputs = self.relu(inputs)
            outputs = self.down_sampling(outputs)

        if self.pool_type is not None:
            outputs = self.conv1(outputs)
            outputs = self.batch_norm1(outputs)

            return outputs


class TextVDCNN(nn.Module):
    """Text VDCNN
    """

    def __init__(self,
                 embeddings_size: int,
                 filters: list,
                 dropout_spatial: float = 0.2,
                 dropout: float = 0.32,
                 activation_conv: Any = "linear",
                 pool_type: str = "max",
                 top_k: int = 2,
                 label_num: int = 2,
                 activate_classify: str = "softmax"):
        """Text VDCNN
        :param embeddings_size: 特征大小
        :param filters: 滤波器配置, eg: [[64, 1], [128, 1], [256, 1], [512, 1]]
        :param dropout_spatial: 空间采样率
        :param dropout: 采样率
        :param activation_conv: 激活函数
        :param pool_type: "max", "k-max", "conv"
        :param top_k: max(k, s * (L-1) / L)
        :param label_num: 类别数
        :param activate_classify: 分类层激活函数
        """
        super(TextVDCNN, self).__init__()
        self.filters = filters
        self.activation_conv = activation_conv
        self.pool_type = pool_type
        self.activate_classify = activate_classify

        self.spatial_dropout = SpatialDropout(p=dropout_spatial)
        self.conv = nn.Conv1d(in_channels=embeddings_size, out_channels=filters[0][0],
                              kernel_size=1, stride=1, padding="same")
        self.relu = nn.ReLU()

        for index, filters_block in enumerate(self.filters):
            for j in range(filters_block[1] - 1):
                setattr(self, f"{index}_{j}_conv_block", ConvolutionalBlock(embeddings_size, filters_block[0]))
            setattr(self, f"{index}_conv_block", ConvolutionalBlock(embeddings_size, filters_block[0]))
            setattr(self, f"{index}_shortcut_pool", ShortcutPool(embeddings_size, filters_block[0],
                                                                 strides=1, pool_type=self.pool_type))

        self.k_max_pool = KMaxPooling(top_k=top_k)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(in_features=embeddings_size, out_features=label_num)

    def forward(self, inputs):
        embeddings = self.spatial_dropout(inputs)
        conv = self.conv(embeddings)
        conv = get_activation(self.activation_conv)(conv)
        block = self.relu(conv)
        block = torch.permute(input=block, dims=[0, 2, 1])

        for index, filters_block in enumerate(self.filters):
            for j in range(filters_block[1] - 1):
                block_mid = getattr(self, f"{index}_{j}_conv_block")(block)
                block = block_mid + block

            # 这里是conv + max-pooling
            block_mid = getattr(self, f"{index}_conv_block")(block)
            block = getattr(self, f"{index}_shortcut_pool")(block, block_mid)

        block = self.k_max_pool(block)
        block = torch.permute(input=block, dims=[0, 2, 1])
        block = self.flatten(block)
        block = self.dropout(block)
        outputs = self.dense(block)
        outputs = get_activation(self.activate_classify)(outputs)
        return outputs
