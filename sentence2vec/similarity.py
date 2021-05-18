#! -*- coding: utf-8 -*-
""" Calculate Similarity
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def weighted_average(emb1, emb2):
    """ 计算特征向量的权重平均分数

    :param emb1: shape = [counts, feature]
    :param emb2: shape = [counts, feature]
    :return:
    """
    inn = (emb1 * emb2).sum(axis=1)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))

    if np.any(emb1norm == 0) or np.any(emb2norm == 0):
        raise RuntimeWarning("内部除数计算出现0值")

    scores = inn / emb1norm / emb2norm

    return scores


def euclidean(emb1, emb2):
    """ 计算特征向量间的欧氏距离

    :param emb1:
    :param emb2:
    :return:
    """