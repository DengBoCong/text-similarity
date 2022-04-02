#! -*- coding: utf-8 -*-
""" Implementation of Simple-Effective-Text-Matching
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
from sim.pytorch.common import get_activation
from typing import Any


class Re2Encoder(nn.Module):
    """RE2 Encoder
    :param embedding_size: feature size
    :param filters_num: filter size
    :param enc_layers: encoder layer num
    :param kernel_size: 卷积核大小
    :param dropout: 采样率
    """

    def __init__(self,
                 embedding_size: int,
                 filters_num: int,
                 enc_layers: int = 2,
                 kernel_size: Any = 3,
                 dropout: float = 0.8):
        super(Re2Encoder, self).__init__()
        self.enc_layers = enc_layers

        for enc_index in range(enc_layers):
            if enc_index > 0:
                setattr(self, f"enc_dropout_{enc_index}", nn.Dropout(p=dropout))
            if enc_index == 0:
                setattr(self, f"enc_conv1d_{enc_index}", nn.Conv1d(
                    in_channels=embedding_size, out_channels=filters_num, kernel_size=kernel_size, padding="same"
                ))
            else:
                setattr(self, f"enc_conv1d_{enc_index}", nn.Conv1d(
                    in_channels=filters_num, out_channels=filters_num, kernel_size=kernel_size, padding="same"
                ))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, mask):
        outputs = inputs
        for enc_index in range(self.enc_layers):
            outputs = mask * outputs
            if enc_index > 0:
                outputs = getattr(self, f"enc_dropout_{enc_index}")(outputs)
            outputs = getattr(self, f"enc_conv1d_{enc_index}")(outputs.permute(0, 2, 1))
            outputs = outputs.permute(0, 2, 1)
            outputs = get_activation("relu")(outputs)

        outputs = self.dropout(outputs)
        return outputs


class Alignment(nn.Module):
    """对齐层"""

    def __init__(self,
                 embedding_size: int,
                 hidden_size: int,
                 dropout: float,
                 align_type: str = "linear",
                 device: Any = None,
                 dtype: Any = None):
        """
        :param embedding_size: feature size
        :param hidden_size: hidden size
        :param dropout: 采样率
        :param align_type: 对齐方式，identity/linear
        :param device: 机器
        :param dtype: 类型
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Alignment, self).__init__()
        self.align_type = align_type

        if align_type == "linear":
            self.linear_dropout1 = nn.Dropout(p=dropout)
            self.linear_dense1 = nn.Linear(in_features=embedding_size + hidden_size, out_features=hidden_size)
            self.linear_dropout2 = nn.Dropout(p=dropout)
            self.linear_dense2 = nn.Linear(in_features=embedding_size + hidden_size, out_features=hidden_size)

        self.temperature = nn.Parameter(nn.init.constant_(torch.empty((), **factory_kwargs),
                                                          math.sqrt(1 / hidden_size)))

    def forward(self, a_inputs, a_mask, b_inputs, b_mask):
        if self.align_type == "identity":
            attention_outputs = torch.matmul(input=a_inputs, other=b_inputs.permute(0, 2, 1)) * self.temperature
        elif self.align_type == "linear":
            a_outputs = self.linear_dropout1(a_inputs)
            a_outputs = self.linear_dense1(a_outputs)
            a_outputs = get_activation("relu")(a_outputs)
            b_outputs = self.linear_dropout2(b_inputs)
            b_outputs = self.linear_dense2(b_outputs)
            b_outputs = get_activation("relu")(b_outputs)
            attention_outputs = torch.matmul(input=a_outputs, other=b_outputs.permute(0, 2, 1)) * self.temperature
        else:
            raise ValueError("`align_type` must be identity or linear")

        attention_mask = torch.matmul(input=a_mask, other=b_mask.permute(0, 2, 1))
        attention_outputs = attention_mask * attention_outputs + (1 - attention_mask) * -1e9
        a_attention = nn.Softmax(dim=1)(attention_outputs)
        b_attention = nn.Softmax(dim=2)(attention_outputs)

        a_feature = torch.matmul(input=a_attention.permute(0, 2, 1), other=a_inputs)
        b_feature = torch.matmul(input=b_attention, other=b_inputs)

        return a_feature, b_feature


class Fusion(nn.Module):
    """Fusion Layer
    """

    def __init__(self,
                 embedding_size: int,
                 hidden_size: int,
                 dropout: float,
                 fusion_type: str = "full"):
        """
        :param embedding_size: feature size
        :param hidden_size: feature size
        :param dropout: 采样率
        :param fusion_type: fusion type，simple/full
        """
        super(Fusion, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fusion_type = fusion_type

        if self.fusion_type == "full":
            self.orig_dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
            self.sub_dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
            self.mul_dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
            self.dropout_layer = nn.Dropout(p=self.dropout)
            self.output_dense = nn.Linear(in_features=hidden_size * 3, out_features=hidden_size)
        elif self.fusion_type == "simple":
            self.dense = nn.Linear(in_features=(embedding_size + hidden_size) * 2, out_features=hidden_size)
        else:
            raise ValueError("`fusion_type` must be full or simple")

    def forward(self, inputs, align_inputs):
        if self.fusion_type == "full":
            outputs = torch.concat(tensors=[
                get_activation("relu")(self.orig_dense(torch.concat(tensors=[inputs, align_inputs], dim=-1))),
                get_activation("relu")(self.sub_dense(torch.concat(tensors=[inputs, inputs - align_inputs], dim=-1))),
                get_activation("relu")(self.mul_dense(torch.concat(tensors=[inputs, inputs * align_inputs], dim=-1)))
            ], dim=-1)
            outputs = self.dropout_layer(outputs)
            outputs = self.output_dense(outputs)
            outputs = get_activation("relu")(outputs)
        else:
            outputs = self.dense(torch.concat(tensors=[inputs, align_inputs], dim=-1))
            outputs = get_activation("relu")(outputs)

        return outputs


class Prediction(nn.Module):
    """Prediction Layer
    """

    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 dropout: float,
                 pred_type: str = "full"):
        """
        :param num_classes: 类别数
        :param hidden_size: feature size
        :param dropout: 采样率
        :param pred_type: prediction type，simple/full/symmetric
        """
        super(Prediction, self).__init__()
        self.pred_type = pred_type
        self.dropout = dropout

        if self.pred_type == "simple":
            in_features = hidden_size * 2
        elif self.pred_type == "full" or self.pred_type == "symmetric":
            in_features = hidden_size * 4
        else:
            raise ValueError("`pred_type` must be simple, full or symmetric")

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dense1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, a_feature, b_feature):
        if self.pred_type == "simple":
            outputs = torch.concat(tensors=[a_feature, b_feature], dim=-1)
        elif self.pred_type == "full":
            outputs = torch.concat(tensors=[a_feature, b_feature, a_feature * b_feature, a_feature - b_feature], dim=-1)
        elif self.pred_type == "symmetric":
            outputs = torch.concat(tensors=[a_feature, b_feature, a_feature * b_feature,
                                            torch.abs((a_feature - b_feature))], dim=-1)
        else:
            raise ValueError("`pred_type` must be simple, full or symmetric")

        outputs = self.dropout1(outputs)
        outputs = self.dense1(outputs)
        outputs = get_activation("relu")(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.dense2(outputs)

        return outputs


class Re2Network(nn.Module):
    """Simple-Effective-Text-Matching
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 block_layer_num: int = 2,
                 enc_layers: int = 2,
                 enc_kernel_size: Any = 3,
                 dropout: float = 0.8,
                 num_classes: int = 2,
                 hidden_size: int = None,
                 connection_args: str = "aug",
                 align_type: str = "linear",
                 fusion_type: str = "full",
                 pred_type: str = "full"):
        """
        :param vocab_size: 词表大小
        :param embedding_size: feature size
        :param block_layer_num: fusion block num
        :param enc_layers: encoder layer num
        :param enc_kernel_size: 卷积核大小
        :param dropout: 采样率
        :param num_classes: 类别数
        :param hidden_size: 隐藏层大小
        :param connection_args: 连接层模式，residual/aug
        :param align_type: 对齐方式，identity/linear
        :param fusion_type: fusion type，simple/full
        :param pred_type: prediction type，simple/full/symmetric
        """
        super(Re2Network, self).__init__()
        self.hidden_size = hidden_size
        if not hidden_size:
            self.hidden_size = embedding_size // 2
        self.block_layer_num = block_layer_num
        self.connection_args = connection_args

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.embeddings_dropout = nn.Dropout(p=dropout)

        self.connection = {
            "residual": self.residual,
            "aug": self.augmented_residual
        }

        for index in range(block_layer_num):
            if index == 0:
                setattr(self, f"re2_encoder_{index}", Re2Encoder(
                    embedding_size=embedding_size, filters_num=hidden_size,
                    enc_layers=enc_layers, kernel_size=enc_kernel_size, dropout=dropout
                ))
                setattr(self, f"alignment_{index}", Alignment(
                    embedding_size=embedding_size, hidden_size=hidden_size, dropout=dropout, align_type=align_type
                ))
                setattr(self, f"fusion_{index}", Fusion(
                    embedding_size=embedding_size, hidden_size=hidden_size, dropout=dropout, fusion_type=fusion_type
                ))
            else:
                setattr(self, f"re2_encoder_{index}", Re2Encoder(
                    embedding_size=embedding_size + hidden_size, filters_num=hidden_size,
                    enc_layers=enc_layers, kernel_size=enc_kernel_size, dropout=dropout
                ))
                setattr(self, f"alignment_{index}", Alignment(
                    embedding_size=embedding_size + hidden_size, hidden_size=hidden_size,
                    dropout=dropout, align_type=align_type
                ))
                setattr(self, f"fusion_{index}", Fusion(
                    embedding_size=embedding_size + hidden_size, hidden_size=hidden_size,
                    dropout=dropout, fusion_type=fusion_type
                ))

        self.prediction = Prediction(num_classes=num_classes, hidden_size=hidden_size,
                                     dropout=dropout, pred_type=pred_type)

    def forward(self, text_a_input_ids, text_b_input_ids):
        text_a_mask = torch.eq(input=text_a_input_ids, other=0).float()[:, :, None]
        text_b_mask = torch.eq(input=text_b_input_ids, other=0).float()[:, :, None]

        a_embeddings = self.embeddings(text_a_input_ids)
        a_outputs = self.embeddings_dropout(a_embeddings)
        b_embeddings = self.embeddings(text_b_input_ids)
        b_outputs = self.embeddings_dropout(b_embeddings)

        a_residual, b_residual = a_outputs, b_outputs

        for index in range(self.block_layer_num):
            if index > 0:
                a_outputs = self.connection[self.connection_args](a_outputs, a_residual, index)
                b_outputs = self.connection[self.connection_args](b_outputs, b_residual, index)
                a_residual, b_residual = a_outputs, b_outputs

            # Encoder
            a_encoder_outputs = getattr(self, f"re2_encoder_{index}")(a_outputs, text_a_mask)
            b_encoder_outputs = getattr(self, f"re2_encoder_{index}")(b_outputs, text_b_mask)

            # cat
            a_outputs = torch.concat(tensors=[a_outputs, a_encoder_outputs], dim=-1)
            b_outputs = torch.concat(tensors=[b_outputs, b_encoder_outputs], dim=-1)

            # alignment
            a_align, b_align = getattr(self, f"alignment_{index}")(a_outputs, text_a_mask, b_outputs, text_b_mask)

            # fusion
            a_outputs = getattr(self, f"fusion_{index}")(a_outputs, a_align)
            b_outputs = getattr(self, f"fusion_{index}")(b_outputs, b_align)

        a_outputs = torch.sum(input=text_a_mask * a_outputs + (1. - text_a_mask) * -1e9, dim=1)
        b_outputs = torch.sum(input=text_b_mask * b_outputs + (1. - text_b_mask) * -1e9, dim=1)

        outputs = self.prediction(a_outputs, b_outputs)

        return outputs

    def residual(self, inputs: Any, res_inputs: Any, _) -> Any:
        """残差"""
        if inputs.shape[-1] != res_inputs.shape[-1]:
            inputs = nn.Linear(in_features=self.hidden_size, out_features=res_inputs.shape[-1])(inputs)
        return (inputs + res_inputs) * math.sqrt(0.5)

    def augmented_residual(self, inputs: Any, res_inputs: Any, index: int) -> Any:
        """增强残差"""
        outputs = inputs
        if index == 1:
            outputs = torch.concat(tensors=[res_inputs, inputs], dim=-1)
        elif index > 1:
            hidden_size = inputs.shape[-1]
            outputs = (res_inputs[:, :, -hidden_size:] + inputs) * math.sqrt(0.5)
            outputs = torch.concat(tensors=[res_inputs[:, :, :-hidden_size], outputs], dim=-1)

        return outputs
