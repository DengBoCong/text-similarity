#! -*- coding: utf-8 -*-
""" Train Pipeline
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import time
from sim.tools.datasets import datasets_generator
from sim.tools.tools import get_dict_string
from sim.tools.tools import ProgressBar
from typing import Any
from typing import NoReturn


class Pipeline(abc.ABC):
    def __init__(self, model: list, loss_metric: Any, accuracy_metric: Any, batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param loss_metric: 损失计算器，必传指标
        :param accuracy_metric: 精度计算器，必传指标
        :param batch_size: batch size
        """
        self.model = model
        self.loss_metric = loss_metric
        self.accuracy_metric = accuracy_metric
        self.batch_size = batch_size

    def train(self, train_file_path: str, valid_file_path: str, epochs: int, optimizer: Any,
              checkpoint: Any, checkpoint_save_freq: int, history=None, *args, **kwargs) -> dict:
        """ Train Module
        :param train_file_path: 已转换为token id的训练数据文件路径
        :param valid_file_path: 已转换为token id的验证数据文件路径
        :param epochs: 训练周期
        :param optimizer: 优化器
        :param checkpoint: 检查点管理器
        :param checkpoint_save_freq: 检查点保存频率
        :param history: 用于保存训练过程中的历史指标数据
        """
        if history is None:
            history = {}

        print("Begin train...")

        progress_bar = ProgressBar()
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))

            start_time = time.time()
            self.loss_metric.reset_states()
            self.accuracy_metric.reset_states()

            for batch, batch_dataset in enumerate(datasets_generator(
                    file_path=train_file_path, batch_size=self.batch_size)):
                if batch == 0:
                    progress_bar.reset(total=batch_dataset[3], num=self.batch_size)

                train_metrics = self._train_step(batch_dataset=batch_dataset, optimizer=optimizer, *args, **kwargs)

                for key, value in train_metrics.items():
                    history[key].append(value)
                progress_bar(current=batch + 1, metrics=get_dict_string(data=train_metrics))

            progress_bar.done(step_time=time.time() - start_time)

            if (epoch + 1) % checkpoint_save_freq == 0:
                checkpoint.save()

                print("Begin valid...")
                self._valid(valid_file_path, progress_bar, history=history, *args, **kwargs)

        print("Train End.")
        self._save_model(*args, **kwargs)
        return history

    def evaluate(self, valid_file_path: str, history=None, *args, **kwargs) -> dict:
        """ 验证模块
        :param valid_file_path: 验证数据文本路径
        :param history: 用于保存evaluate过程中的历史指标数据
        :return: 返回历史指标数据
        """
        print("Begin evaluate...")
        if history is None:
            history = {}

        progress_bar = ProgressBar()
        self._valid(valid_file_path, progress_bar, history=history, *args, **kwargs)

        print("Evaluate end.")
        return history

    def _valid(self, valid_file_path: str, progress_bar: ProgressBar, history=None, *args, **kwargs) -> NoReturn:
        """ 验证模块
        :param valid_file_path: 验证数据文本路径
        :param progress_bar: 进度工具
        :param history: 用于保存evaluate过程中的历史指标数据
        :return: 返回历史指标数据
        """
        print("Begin evaluate...")

        valid_start_time = time.time()
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()

        for valid_batch, valid_batch_dataset in enumerate(datasets_generator(
                file_path=valid_file_path, batch_size=self.batch_size)):
            if valid_batch == 0:
                progress_bar.reset(total=valid_batch_dataset[3], num=self.batch_size)

            valid_metrics = self._valid_step(dataset=valid_batch_dataset, *args, **kwargs)

            for key, value in valid_metrics.items():
                history[key].append(value)

        progress_bar.done(step_time=time.time() - valid_start_time)

    @abc.abstractmethod
    def _train_step(self, batch_dataset: tuple, optimizer: Any, *args, **kwargs) -> dict:
        """该方法用于定于训练步中，模型实际训练的核心代码（在train方法中使用）
            batch_dataset=[input1,input2,label,steps]，输出必须满足输出dict
        Note:
            a): 返回所得指标字典
            b): batch_dataset、optimizer为模型训练必需
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def _valid_step(self, dataset: tuple, *args, **kwargs) -> dict:
        """ 该方法用于定义验证模型逻辑
            batch_dataset=[input1,input2,label,steps]，输出必须满足输出dict
        Note:
            a): 返回所得指标字典
            b): dataset为模型验证必需
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def _save_model(self, *args, **kwargs) -> NoReturn:
        """ 将模型保存为SaveModel格式
        Note:
            如果不在train之后保存SaveModel，子类继承实现这个方法时，直接pass即可
        """

        raise NotImplementedError("Must be implemented in subclasses.")
