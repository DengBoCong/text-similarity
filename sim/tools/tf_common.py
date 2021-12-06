#! -*- coding: utf-8 -*-
""" Tensorflow Common Tools
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Any


def load_checkpoint(checkpoint_dir: str, execute_type: str, checkpoint_save_size: int, model: tf.keras.Model = None,
                    encoder: tf.keras.Model = None, decoder: tf.keras.Model = None) -> tf.train.CheckpointManager:
    """加载检查点，同时支持Encoder-Decoder结构加载，两种类型的模型二者只能传其一
    :param checkpoint_dir: 检查点保存目录
    :param execute_type: 执行类型
    :param checkpoint_save_size: 检查点最大保存数量
    :param model: 传入的模型
    :param encoder: 传入的Encoder模型
    :param decoder: 传入的Decoder模型
    """
    if model is not None:
        checkpoint = tf.train.Checkpoint(model=model)
    elif encoder is not None and decoder is not None:
        checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    else:
        raise ValueError("Create checkpoint error")

    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir,
                                                    max_to_keep=checkpoint_save_size)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    else:
        if execute_type != "train" and execute_type != "pre_treat":
            raise FileNotFoundError("Not found checkpoint file")

    return checkpoint_manager


# 定义相关的损失函数
def contrastive_loss(ew: Any, label: Any, m: float):
    """
    :param ew: Embedding向量之间的度量
    :param label: 样本句子的标签
    :param m: 负样本控制阈值
    :return:
    """
    l_1 = 0.25 * (1.0 - ew) * (1.0 - ew)
    l_0 = tf.where(condition=ew < m * tf.ones_like(input=ew), x=tf.fill(dims=ew.shape, value=0), y=ew) * tf.where(
        condition=ew < m * tf.ones_like(input=ew), x=tf.fill(dims=ew.shape, value=0), y=ew)

    loss = label * 1.0 * l_1 + (1 - label) * 1.0 * l_0

    return loss.sum()
