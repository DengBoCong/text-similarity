#! -*- coding: utf-8 -*-
""" Train Pipeline
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random
import tensorflow as tf
import time
from sim.tools.datasets import datasets_generator
from sim.tools.tools import get_dict_string
from sim.tools.tools import ProgressBar
from typing import Any
from typing import NoReturn


class Pipeline(object):
    def __init__(self, model: list, loss_metric: tf.keras.metrics.Mean,
                 accuracy_metric: tf.keras.metrics.SparseCategoricalAccuracy, batch_size: int):
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

    def train(self, train_step: Any, valid_step: Any, save_model: Any, seed: int, train_file_path: str,
              valid_file_path: str, epochs: int, optimizer: tf.optimizers.Adam,
              checkpoint: tf.train.CheckpointManager, checkpoint_save_freq: int, history=None, **kwargs) -> dict:
        """ Train Module
        :param train_step: 自定义训练步方法，[batch_dataset,optimizer]为必接收的参数，
                           batch_dataset=[input1,input2,label,steps]，输出必须满足输出dict
        :param valid_step: 自定义验证步方法，[dataset]为必接收的参数，batch_dataset=[input1,input2,label,steps]，输出必须满足输出dict
        :param save_model: 训练完毕后，进行模型保存方法
        :param seed: 随机种子
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

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

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

                train_metrics = train_step(batch_dataset=batch_dataset, optimizer=optimizer, **kwargs)

                for key, value in train_metrics.items():
                    history[key].append(value)
                progress_bar(current=batch + 1, metrics=get_dict_string(data=train_metrics))

            progress_bar.done(step_time=time.time() - start_time)

            if (epoch + 1) % checkpoint_save_freq == 0:
                checkpoint.save()

                print("Begin valid...")
                self._valid(valid_file_path, valid_step, progress_bar, history=history, **kwargs)

        print("Train End.")
        save_model(**kwargs)
        return history

    def evaluate(self, valid_file_path: str, valid_step: Any, history=None, **kwargs) -> dict:
        """ 验证模块
        :param valid_file_path: 验证数据文本路径
        :param valid_step: 自定义验证步方法，[dataset]为必接收的参数，batch_dataset=[input1,input2,label,steps]，输出必须满足输出dict
        :param history: 用于保存evaluate过程中的历史指标数据
        :return: 返回历史指标数据
        """
        print("Begin evaluate...")
        if history is None:
            history = {}

        progress_bar = ProgressBar()
        self._valid(valid_file_path, valid_step, progress_bar, history=history, **kwargs)

        print("Evaluate end.")
        return history

    def _valid(self, valid_file_path: str, valid_step: Any,
               progress_bar: ProgressBar, history=None, **kwargs) -> NoReturn:
        """ 验证模块
        :param valid_file_path: 验证数据文本路径
        :param valid_step: 自定义验证步方法，[dataset]为必接收的参数，batch_dataset=[input1,input2,label,steps]，输出必须满足输出dict
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

            valid_metrics = valid_step(dataset=valid_batch_dataset, **kwargs)

            for key, value in valid_metrics.items():
                history[key].append(value)

        progress_bar.done(step_time=time.time() - valid_start_time)

    def inference(self, request: str, beam_size: int, start_sign: str = "<start>", end_sign: str = "<end>") -> Any:
        """ 对话推断模块
        :param request: 输入句子
        :param beam_size: beam大小
        :param start_sign: 句子开始标记
        :param end_sign: 句子结束标记
        :return: 返回历史指标数据
        """
        pass
