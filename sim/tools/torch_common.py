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
            print("没有检查点，请先执行train模式")
            exit(0)
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
