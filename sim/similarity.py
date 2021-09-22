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
from typing import Any


def euclidean_dist(emb1: np.ndarray, emb2: np.ndarray, axis=-1) -> Any:
    """ 计算特征向量间的欧氏距离
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: 计算维度
    :return: 欧式距离度量
    """
    return np.sqrt(np.square(emb1 - emb2).sum(axis=axis))


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray, dist: bool = False, axis: int = -1) -> Any:
    """ 计算特征向量间的余弦相似度
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param dist: 是否返回余弦距离(归一化后)，取值范围[0, 1]
    :param axis: 计算维度
    :return: 欧式距离度量
    """
    mod = np.linalg.norm(emb1, axis=axis) * np.linalg.norm(emb2, axis=axis)
    if np.all(mod == 0):
        raise RuntimeError("cosine similarity divisor is zero")
    cos = np.sum(emb1 * emb2, axis=axis, dtype=float) / mod

    return (1 - cos) / 2.0 if dist else cos


def manhattan_dist(emb1: np.ndarray, emb2: np.ndarray, axis: int = -1) -> Any:
    """ 计算特征向量间的曼哈顿距离
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: 计算维度
    :return: 曼哈顿距离
    """
    return np.sum(np.abs(emb1 - emb2), axis=axis)


def minkowsk_dist(emb1: np.ndarray, emb2: np.ndarray, p: int, axis: int = -1) -> Any:
    """ 计算特征向量间的闵可夫斯基距离，注意开方正负
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param p: 范数
    :param axis: 计算维度
    :return: 闵可夫斯基距离
    """
    tmp = np.sum(np.power(emb1 - emb2, p), axis=axis)
    return np.power(tmp, 1 / p)


def hamming_dist(emb1: np.ndarray, emb2: np.ndarray, axis: int = -1):
    """ 计算特征向量的汉明距离

    :param emb1: shape = [feature,]
    :param emb2: shape = [feature,]
    :param axis: 计算维度
    :return: 汉明距离
    """
    if len(emb1.shape) > 1 or len(emb2.shape) > 1:
        raise RuntimeError("the shape of emb1 and emb2 must be [feature,]")

    avg1, avg2 = np.mean(emb1, axis=axis), np.mean(emb2, axis=axis)
    binary1, binary2 = np.where(emb1 < avg1, 0, 1), np.where(emb2 < avg2, 0, 1)
    return len(np.nonzero(binary1 - binary2)[0])


def jaccard_similarity(emb1: np.ndarray, emb2: np.ndarray, axis: int = -1):
    """ 计算特征向量的Jaccard系数

    :param emb1: shape = [feature,]
    :param emb2: shape = [feature,]
    :param axis: 计算维度
    :return: Jaccard 洗漱
    """
    # print(emb1 != emb2)
    # print(emb1 != 0)
    # print(emb2 != 0)
    # print(np.double(np.bitwise_and((emb1 != emb2), np.bitwise_or(emb1 != 0, emb2 != 0)).sum()))
    # exit(0)
    up = np.double(np.bitwise_and((emb1 != emb2), np.bitwise_or(emb1 != 0, emb2 != 0)).sum())
    down = np.double(np.bitwise_or(emb1 != 0, emb2 != 0).sum())
    d1 = (up / down)
    return d1


def _weighted_average(emb1, emb2, axis=-1):
    """ 计算特征向量的权重平均分数

    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: 计算维度
    :return: 余弦相似度分数，shape = [...,]
    """
    inn = (emb1 * emb2).sum(axis=axis)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=axis))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=axis))

    if np.any(emb1norm == 0) or np.any(emb2norm == 0):
        raise RuntimeWarning(" divisor is zero")

    scores = inn / emb1norm / emb2norm

    return scores

# 编辑距离
