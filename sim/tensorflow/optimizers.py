#! -*- coding: utf-8 -*-
""" Custom Optimizers Common
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras


class PiecewiseLinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    """分段线性学习率
    """

    def __init__(self, boundaries: list, values: list, start: float = 0.):
        """ 如boundaries=[10000, 11000], values=[1.0, 0.1]
            指的是0-10000步，从start均匀增加至1.，10000-11000，从1.均匀降低到0.1
            11000以后保持在0.1不变
        :param boundaries: 分段范围, len=2
        :param values: 对应学习率, len=2
        :param start: 从start开始
        """
        assert start <= values[0] <= 1. and values[0] >= values[1]
        assert boundaries[0] < boundaries[1]
        self.boundaries = boundaries
        self.values = values
        self.start = start

    def __call__(self, step):
        if step <= self.boundaries[0]:
            slope = (self.values[0] - self.start) / self.boundaries[0]
            return self.start + step * slope
        elif self.boundaries[0] < step <= self.boundaries[1]:
            slope = (self.values[0] - self.values[1]) / (self.boundaries[1] - self.boundaries[0])
            return self.values[0] - (step - self.boundaries[0]) * slope
        else:
            return self.values[1]

    def get_config(self):
        config = {
            "start": self.start,
            "boundaries": self.boundaries,
            "values": self.values
        }
        base_config = super(PiecewiseLinearDecay, self).get_config()
        config.update(base_config)
        return config
