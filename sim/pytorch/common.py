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
from collections import OrderedDict
from typing import Any
from typing import NoReturn


def load_embeddings(embeddings: torch.Tensor, keep_tokens: torch.Tensor = None, compound_tokens: list = None) -> Any:
    """给加载与训练权重时，token不一致进行额外处理用
    :param embeddings: 原全量embedding
    :param keep_tokens: 要保留的词ID列表
    :param compound_tokens: 扩展Embedding
    """
    if keep_tokens is not None:
        embeddings = embeddings.index_select(dim=0, index=keep_tokens)

    if compound_tokens is not None:
        ext_embeddings = []
        for item in compound_tokens:
            if isinstance(item, list):
                item = (item, [1] * len(item))

            ext_embeddings.append(np.average(embeddings.index_select(dim=0, index=torch.tensor(item[0])), 0, item[1]))

        embeddings = np.concatenate(arrays=[embeddings, ext_embeddings], axis=0)

    return embeddings


def load_bert_weights(model_file_path: str,
                      model: nn.Module,
                      mapping: dict,
                      keep_tokens: torch.Tensor = None,
                      compound_tokens: list = None) -> OrderedDict:
    """根据mapping从权重文件中加载bert权重
    :param model_file_path: 权重文件路径
    :param model: 模型
    :param mapping: 权重映射表
    :param keep_tokens: 要保留的词ID列表
    :param compound_tokens: 扩展Embedding
    """
    success_load_count = 0
    model_state_dict, pretrain_state_dict = model.state_dict(), torch.load(model_file_path)
    for k, v in model_state_dict.items():
        if k in mapping and mapping[k] in pretrain_state_dict:
            if mapping[k] in [
                "bert.embeddings.word_embeddings.weight",
                "cls.predictions.decoder.weight",
                "cls.predictions.bias"
            ]:
                model_state_dict[k] = load_embeddings(pretrain_state_dict[mapping[k]], keep_tokens, compound_tokens)
                success_load_count += 1
            else:
                assert model_state_dict[k].shape == pretrain_state_dict[mapping[k]].shape
                model_state_dict[k] = pretrain_state_dict[mapping[k]]
                success_load_count += 1

    # 这里做一个权重成功加载的数量提示
    print(f"success load weights count: {success_load_count}")

    return model_state_dict


def load_weights(model_file_path: str,
                 model: nn.Module,
                 mapping: dict = None) -> OrderedDict:
    """根据mapping从权重文件中加载模型权重
    :param model_file_path: 权重文件路径
    :param model: 模型
    :param mapping: 权重映射表
    """
    success_load_count = 0
    model_state_dict, pretrain_state_dict = model.state_dict(), torch.load(model_file_path)
    for k, v in model_state_dict.items():
        if mapping and k in mapping and mapping[k] in pretrain_state_dict:
            assert model_state_dict[k].shape == pretrain_state_dict[mapping[k]].shape
            model_state_dict[k] = pretrain_state_dict[mapping[k]]
            success_load_count += 1
        elif k in pretrain_state_dict:
            assert model_state_dict[k].shape == pretrain_state_dict[k].shape
            model_state_dict[k] = pretrain_state_dict[k]
            success_load_count += 1
        else:
            print(f"`{k}` not in weights mapping, and ignore")

    # 这里做一个权重成功加载的数量提示
    print(f"success load weights count: {success_load_count}")

    return model_state_dict


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
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint")
        version = 1
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                info = json.load(file)
                version = info["version"] + 1

        model_dict = {}
        if self.model is not None:
            model_dict["model_state_dict"] = self.model.state_dict()
        model_dict["optimizer_state_dict"] = self.optimizer.state_dict()

        model_checkpoint_path = "checkpoint-{}.bin".format(version)
        torch.save(model_dict, os.path.join(self.checkpoint_dir, model_checkpoint_path))
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

    def _truncated_norm(tensor: torch.Tensor):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.detach().copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.detach().mul_(stddev).add_(mean)
            return tensor

    return _truncated_norm


def sinusoidal_init_(position: int, depth: int) -> NoReturn:
    """Sin-Cos位置向量初始化器
    :param position: 位置大小
    :param depth: 位置嵌入大小
    """

    def _sinusoidal_init(tensor: torch.Tensor):
        with torch.no_grad():
            pos = np.arange(position)[:, np.newaxis]
            index = np.arange(depth)[np.newaxis, :]

            angle_rates = 1 / np.power(10000, (2 * (index // 2)) / np.float32(depth))
            angle_rads = pos * angle_rates
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

            tensor.detach().copy_(torch.from_numpy(angle_rads))
            return tensor

    return _sinusoidal_init


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


def dot_product_attention(query: Any,
                          key: Any,
                          value: Any,
                          depth: int,
                          dropout: float,
                          mask: Any = None) -> tuple:
    """通用点乘注意力计算
    :param query: (..., seq_len_q, depth)
    :param key: (..., seq_len_k, depth)
    :param value: (..., seq_len_v, depth_v)
    :param depth: 分头之后维度大小
    :param dropout: 注意力dropout
    :param mask: float, (..., seq_len_q, seq_len_k)
    """
    attention_scores = torch.matmul(input=query, other=key.permute(0, 2, 1))
    attention_scores = attention_scores / torch.sqrt(input=torch.tensor(data=depth))

    if mask is not None:
        attention_scores += (mask * -1e9)

    attention_weights = torch.softmax(input=attention_scores, dim=-1)
    attention_weights = nn.Dropout(p=dropout)(attention_weights)
    context_layer = torch.matmul(input=attention_weights, other=value)

    return context_layer, attention_weights


def swish(x):
    return x * torch.sigmoid(x)


def linear_act(x):
    return x


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
        "tanh": torch.tanh,
        "softmax": F.softmax,
        "swish": swish,
        "linear": linear_act
    }

    if identifier not in activations:
        raise ValueError(f"{identifier} not such activation")

    return activations[identifier]
