#! -*- coding: utf-8 -*-
""" Pytorch Common Tools
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from typing import NoReturn


def get_activation(identifier: str):
    """获取激活函数
    """
    activations = {
        "gelu": F.gelu,
        "glu": F.glu,
        "relu": F.relu,
        "elu": F.elu,
        "hardtanh": F.hardtanh,
        "relu6": F.relu6,
        "selu": F.selu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": F.sigmoid,
        "gumbel_softmax": F.gumbel_softmax,
        "tanh": F.tanh
    }

    if identifier not in activations:
        raise ValueError(f"{identifier} not such activation")

    return activations[identifier]


class Checkpoint(object):
    def __init__(self, checkpoint_dir: str, optimizer: torch.optim.Optimizer = None, model: torch.nn.Module = None):
        """
        :param checkpoint_dir: 检查点保存路径
        :param optimizer: 优化器
        :param model: 模型
        """
        super(Checkpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = optimizer
        self.model = model

    def save(self) -> NoReturn:
        """ 保存模型检查点
        :return: 无返回值
        """
        checkpoint_path = self.checkpoint_dir + "checkpoint"
        version = 1
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                info = json.load(file)
                version = info["version"] + 1

        model_dict = {}
        if self.model is not None:
            model_dict["model_state_dict"] = self.model.state_dict()
        model_dict["optimizer_state_dict"] = self.optimizer.state_dict()

        model_checkpoint_path = "checkpoint-{}.pth".format(version)
        torch.save(model_dict, self.checkpoint_dir + model_checkpoint_path)
        with open(checkpoint_path, "w", encoding="utf-8") as file:
            file.write(json.dumps({
                "version": version,
                "model_checkpoint_path": model_checkpoint_path,
                "last_preserved_timestamp": time.time()
            }))

    def load(self, execute_type: str) -> tuple:
        """加载检查点恢复模型，同时支持Encoder-Decoder结构加载
        :param execute_type: 执行类型
        :return: 恢复的各模型检查点细节
        """
        checkpoint_path = self.checkpoint_dir + "checkpoint"

        if not os.path.exists(checkpoint_path) and execute_type != "train" and execute_type != "pre_treat":
            raise FileNotFoundError("checkpoint_path not found, please execute train first")
        elif not os.path.exists(checkpoint_path):
            return self.model, self.optimizer

        with open(checkpoint_path, "r", encoding="utf-8") as file:
            checkpoint_info = json.load(file)

        model_checkpoint_path = self.checkpoint_dir + checkpoint_info["model_checkpoint_path"]

        checkpoint = torch.load(model_checkpoint_path)
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return self.model, self.optimizer


def set_seed(manual_seed):
    """ 固定随机种子
    :manual_seed: 手动指定种子
    :return: None
    """
    random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


# 定义相关的损失函数
class ContrastiveLoss(nn.Module):
    """ 对比损失函数"""

    def __init__(self) -> NoReturn:
        super(ContrastiveLoss, self).__init__()

    def forward(self, ew: Any, label: Any, m: float):
        """
        :param ew: Embedding向量之间的度量
        :param label: 样本句子的标签
        :param m: 负样本控制阈值
        :return:
        """
        l_1 = 0.25 * (1.0 - ew) * (1.0 - ew)
        l_0 = torch.where(ew < m * torch.ones_like(ew), torch.full_like(ew, 0), ew) * torch.where(
            ew < m * torch.ones_like(ew), torch.full_like(ew, 0), ew)

        loss = label * 1.0 * l_1 + (1 - label) * 1.0 * l_0

        return loss.sum()


def truncated_normal_(mean: float = 0.0, stddev: float = 0.02) -> Any:
    """截尾正态分布
    :param mean: 均值
    :param stddev: 标准差
    """

    def _truncated_norm(tensor: Any):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.detach().copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.detach().mul_(stddev).add_(mean)
            return tensor

    return _truncated_norm


def scaled_dot_product_attention(query: Any,
                                 key: Any,
                                 value: Any,
                                 batch_size: int,
                                 num_heads: int,
                                 attention_head_size: int,
                                 dropout: float,
                                 mask: Any = None,
                                 pos_type: str = None,
                                 pos_ids: Any = None) -> tuple:
    """点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param batch_size: batch size
    :param num_heads: head num
    :param attention_head_size: 分头之后维度大小
    :param dropout: 注意力dropout
    :param mask: float, (..., seq_len_q, seq_len_k)
    :param pos_type: 指定位置编码种类，现支持经典的相对位置编码: "typical_relation"
    :param pos_ids: 位置编码
    """
    attention_scores = torch.matmul(input=query, other=key.permute(0, 1, 3, 2))
    # 处理位置编码
    if pos_type == "typical_relation":
        attention_scores = attention_scores + torch.einsum("bhjd,kjd->bhjk", query, pos_ids)
    attention_scores = attention_scores / torch.sqrt(input=torch.tensor(data=attention_head_size))

    if mask is not None:
        attention_scores += (mask * -1e9)

    attention_weights = torch.softmax(input=attention_scores, dim=-1)
    attention_weights = nn.Dropout(p=dropout)(attention_weights)

    context_layer = torch.matmul(input=attention_weights, other=value)
    if pos_type == "typical_relation":
        context_layer = context_layer + torch.einsum("bhjk,jkd->bhjd", attention_weights, pos_ids)
    context_layer = context_layer.permute(0, 2, 1, 3)
    context_layer = torch.reshape(input=context_layer, shape=(batch_size, -1, attention_head_size * num_heads))

    return context_layer, attention_weights
