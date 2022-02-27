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
from sim.tools.data_processor.data_format import DataGenerator
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_dict_string
from sim.tools.tools import get_logger
from sim.tools.tools import ProgressBar
from typing import Any
from typing import NoReturn

logger = get_logger(name="pipeline", file_path=RUNTIME_LOG_FILE_PATH)


class Pipeline(abc.ABC):
    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


class NormalPipeline(Pipeline):
    def __init__(self, model: list, batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        """
        self.model = model
        self.batch_size = batch_size

    def train(self,
              train_generator: DataGenerator,
              valid_generator: DataGenerator,
              epochs: int,
              optimizer: Any,
              checkpoint: Any,
              checkpoint_save_freq: int,
              history: dict = None, *args, **kwargs) -> dict:
        """ Train Module
        :param train_generator: 训练数据生成器
        :param valid_generator: 验证数据生成器
        :param epochs: 训练周期
        :param optimizer: 优化器
        :param checkpoint: 检查点管理器
        :param checkpoint_save_freq: 检查点保存频率
        :param history: 用于保存训练过程中的历史指标数据
        """
        if history is None:
            history = {}

        logger.info("Begin train")

        progress_bar = ProgressBar()
        for epoch in range(epochs):
            logger.info("Epoch {}/{}".format(epoch + 1, epochs))

            start_time = time.time()

            for batch, batch_dataset in enumerate(train_generator):
                if batch == 0:
                    progress_bar.reset(total=train_generator.steps, num=self.batch_size)

                train_metrics = self._train_step(batch_dataset=batch_dataset, optimizer=optimizer, *args, **kwargs)

                metrics = {}
                for key, value in train_metrics.items():
                    history[key].append(value)
                    metrics[f"avg_{key}"] = sum(history[key]) / len(history[key])
                train_metrics.update(metrics)
                progress_bar(current=batch + 1, metrics=get_dict_string(data=train_metrics))

            logger.info(progress_bar.done(step_time=time.time() - start_time))

            if (epoch + 1) % checkpoint_save_freq == 0:
                checkpoint.save()

                logger.info("Begin valid")
                self._valid(progress_bar, data_generator=valid_generator, history=history, *args, **kwargs)

        logger.info("Train end")
        self._save_model(*args, **kwargs)
        return history

    def evaluate(self, data_generator: DataGenerator, history=None, *args, **kwargs) -> dict:
        """ 验证模块
        :param data_generator: 数据生成器
        :param history: 用于保存evaluate过程中的历史指标数据
        :return: 返回历史指标数据
        """
        logger.info("Begin evaluate")
        if history is None:
            history = {}

        progress_bar = ProgressBar()
        self._valid(progress_bar, data_generator=data_generator, history=history, *args, **kwargs)

        logger.info("Evaluate end")
        return history

    def _valid(self,
               progress_bar: ProgressBar,
               data_generator: DataGenerator,
               history: dict = None, *args, **kwargs) -> NoReturn:
        """ 验证模块
        :param progress_bar: 进度工具
        :param data_generator: 数据生成器，最后一个元素必须是step数
        :param history: 用于保存evaluate过程中的历史指标数据
        :return: 返回历史指标数据
        """
        logger.info("Begin evaluate")

        valid_start_time = time.time()

        for valid_batch, batch_dataset in enumerate(data_generator):
            if valid_batch == 0:
                progress_bar.reset(total=data_generator.steps, num=self.batch_size)

            valid_metrics = self._valid_step(batch_dataset=batch_dataset, *args, **kwargs)

            metrics = {}
            for key, value in valid_metrics.items():
                history[key].append(value)
                metrics[f"avg_{key}"] = sum(history[key]) / len(history[key])
            valid_metrics.update(metrics)

            progress_bar(current=valid_batch + 1, metrics=get_dict_string(data=valid_metrics))

        progress_bar.done(step_time=time.time() - valid_start_time)

    @abc.abstractmethod
    def _train_step(self, batch_dataset: dict, optimizer: Any, *args, **kwargs) -> dict:
        """该方法用于定于训练步中，模型实际训练的核心代码（在train方法中使用）
            batch_dataset输出必须满足输出dict
        Note:
            a): 返回所得指标字典
            b): batch_dataset、optimizer为模型训练必需
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 该方法用于定义验证模型逻辑
            batch_dataset输出必须满足输出dict
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
