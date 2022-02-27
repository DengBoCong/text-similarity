#! -*- coding: utf-8 -*-
""" Tensorflow Pipeline
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras
from sim.tools.pipeline import NormalPipeline
from typing import Any
from typing import NoReturn


class TextPairPipeline(NormalPipeline):
    def __init__(self, model: list, batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        """
        super(TextPairPipeline, self).__init__(model, batch_size)

    def _metrics(self, y_true: Any, y_pred: Any):
        """指标计算
        :param y_true: 真实标签
        :param y_pred: 预测值
        """
        loss = keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        accuracy = keras.metrics.SparseCategoricalAccuracy()(tf.expand_dims(y_true, axis=1), y_pred)

        return loss, accuracy

    @tf.function(autograph=True)
    def _train_step(self, batch_dataset: dict, optimizer: keras.optimizers.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        with tf.GradientTape() as tape:
            outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])
            loss, accuracy = self._metrics(batch_dataset["labels"], outputs)

        variables = self.model[0].trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return {"t_loss": loss, "t_acc": accuracy}

    @tf.function(autograph=True)
    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 验证步
        :param batch_dataset: 验证步的当前batch数据
        """
        outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])
        loss, accuracy = self._metrics(batch_dataset["labels"], outputs)

        return {"v_loss": loss, "v_acc": accuracy}

    def _save_model(self, *args, **kwargs) -> NoReturn:
        pass
