#! -*- coding: utf-8 -*-
""" Data Format
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from typing import Any
from typing import NoReturn


class ClassificationInputSample(object):
    """seq分类的输入样本"""

    def __init__(self, guid: Any, text_a: str, text_b: str = None, label: str = None) -> NoReturn:
        """构建ClassificationInputSample
        :param guid: 样本唯一ID
        :param text_a: 原始seq文本a
        :param text_b: 原始seq文本b，可选，针对文本对分类任务
        :param label: 样本标签，提供给train/dev，test可选
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class BertInputSample(object):
    """Bert数据的feature集合"""

    def __init__(self, guid: Any, input_ids: list, input_mask: list, segment_ids: list, label: list) -> NoReturn:
        """构建BertInputSample
        :param guid: 样本唯一ID
        :param input_ids: input ids
        :param input_mask: input mask
        :param segment_ids: segment ids
        :param label: label
        """
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class DataGenerator(object):
    """数据生成器基类
    """

    def __init__(self, data: Any, batch_size: int, buffer_size: int = None, steps: int = None, random: bool = True):
        """
        :param data: 数据
        :param batch_size: batch size
        :param buffer_size: 缓冲大小
        :param steps: 总步数
        :param random: 是否打乱数据
        """

        self.data = data
        self.batch_size = batch_size
        self.random = random
        if steps is not None:
            self.steps = steps
        elif hasattr(self.data, "__len__"):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            raise ValueError("pass `steps` or self.data implemented __iter__")

        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def __iter__(self):
        """实现数据生成器
        """
        raise NotImplementedError("__iter__ must be implemented")


class NormalDataGenerator(DataGenerator):
    """常用数据生成器
    """

    def __init__(self, data: Any, batch_size: int, buffer_size: int = None, steps: int = None, random: bool = True):
        """
        :param data: 数据
        :param batch_size: batch size
        :param buffer_size: 缓冲大小
        :param steps: 总步数
        """
        super(NormalDataGenerator, self).__init__(data, batch_size, buffer_size, steps, random)

    def __iter__(self):
        if self.random:
            np.random.shuffle(self.data)

        for i in range(self.steps):
            input1, input2, label = [], [], []
            for sample in self.data[i:i + self.batch_size]:
                sample = sample.split("\t")
                input1.append(list(map(int, sample[0].split(" "))))
                input2.append(list(map(int, sample[1].split(" "))))
                label.append(int(sample[2]) if len(sample) == 3 else 0)

            yield {"inputs1": np.asarray(input1), "inputs2": np.asarray(input2), "labels": np.asarray(label)}


class SimCSEDataGenerator(DataGenerator):
    """SimCSE数据生成器
    """

    def __init__(self, data: Any, batch_size: int, buffer_size: int = None, steps: int = None, random: bool = True):
        """
        :param data: 数据
        :param batch_size: batch size
        :param buffer_size: 缓冲大小
        :param steps: 总步数
        """
        super(SimCSEDataGenerator, self).__init__(data, batch_size, buffer_size, steps, random)

    def __iter__(self):
        if self.random:
            np.random.shuffle(self.data)

        for i in range(self.steps):
            input1 = []
            for sample in self.data[i:i + self.batch_size]:
                token_ids = list(map(int, sample.split("\t")[0].split(" ")))
                input1.append(token_ids)
                input1.append(token_ids)

            input1 = np.asarray(input1)

            yield {"inputs1": input1, "inputs2": np.zeros_like(input1), "labels": np.zeros_like(input1[:, :1])}
