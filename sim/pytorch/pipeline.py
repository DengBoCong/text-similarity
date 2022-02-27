#! -*- coding: utf-8 -*-
""" Pytorch Pipeline
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from sim.tools.pipeline import NormalPipeline
from typing import Any
from typing import NoReturn


class TextPairPipeline(NormalPipeline):
    def __init__(self, model: list, batch_size: int, device: Any, inp_dtype: Any, lab_dtype: Any):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        :param device: 设备
        :param inp_dtype: 输入类型
        :param lab_dtype: 标签类型
        """
        super(TextPairPipeline, self).__init__(model, batch_size)
        self.device = device
        self.inp_dtype = inp_dtype
        self.lab_dtype = lab_dtype

    def _metrics(self, y_true: Any, y_pred: Any):
        """指标计算
        :param y_true: 真实标签
        :param y_pred: 预测值
        """
        loss = nn.CrossEntropyLoss()(y_pred, y_true)
        accuracy = torch.eq(torch.argmax(y_pred, dim=-1), y_true).sum(dim=-1).div(self.batch_size)

        return loss, accuracy

    def _train_step(self, batch_dataset: dict, optimizer: torch.optim.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        inputs1 = torch.from_numpy(batch_dataset["inputs1"]).type(self.inp_dtype).to(self.device)
        inputs2 = torch.from_numpy(batch_dataset["inputs2"]).type(self.inp_dtype).to(self.device)
        labels = torch.from_numpy(batch_dataset["labels"]).type(self.lab_dtype).to(self.device)
        outputs = self.model[0](inputs1, inputs2)
        loss, accuracy = self._metrics(labels, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"t_loss": loss, "t_acc": accuracy}

    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 验证步
        :param batch_dataset: 验证步的当前batch数据
        """
        inputs1 = torch.from_numpy(batch_dataset["inputs1"]).type(self.inp_dtype).to(self.device)
        inputs2 = torch.from_numpy(batch_dataset["inputs2"]).type(self.inp_dtype).to(self.device)
        labels = torch.from_numpy(batch_dataset["labels"]).type(self.lab_dtype).to(self.device)
        with torch.no_grad():
            outputs = self.model[0](inputs1, inputs2)
            loss, accuracy = self._metrics(labels, outputs)

        return {"v_loss": loss, "v_acc": accuracy}

    def _save_model(self, *args, **kwargs) -> NoReturn:
        pass
