#! -*- coding: utf-8 -*-
""" Pytorch Common Tools
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import torch
import torch.nn as nn
from typing import Any
from typing import NoReturn


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
