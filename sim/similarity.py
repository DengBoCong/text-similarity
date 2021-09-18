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
        raise RuntimeWarning("内部除数计算出现0值")

    scores = inn / emb1norm / emb2norm

    return scores


def euclidean_dist(emb1, emb2, axis=-1):
    """ 计算特征向量间的欧氏距离

    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: 计算维度
    :return: 欧式距离度量，shape = [...,]
    """
    dist = np.sqrt(np.square(emb1 - emb2).sum(axis=axis))

    return dist


def cosine_similarity(emb1, emb2, dist=False, axis=-1):
    """ 计算特征向量间的余弦相似度

    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param dist: 是否返回余弦距离(归一化后)，取值范围[0,2]
    :param axis: 计算维度
    :return: 欧式距离度量，shape = [...,]
    """
    mod = np.linalg.norm(emb1, axis=axis) * np.linalg.norm(emb2, axis=axis)
    if np.all(mod == 0):
        raise RuntimeWarning("内部除数计算出现0值")
    cos = (np.sum(emb1 * emb2, axis=axis, dtype=float) / mod)

    return (1 - cos) / 2.0 if dist else cos


# def jaccard_similarity(emb1, emb2):
#     up = np.double(np.bitwise_and((x != y), np.bitwise_or(x != 0, y != 0)).sum())
#     down = np.double(np.bitwise_or(x != 0, y != 0).sum())
#     d1 = (up / down)
#     return d1


def hamming_dist(emb1, emb2):
    """ 计算特征向量的汉明距离

    :param emb1: shape = [feature,]
    :param emb2: shape = [feature,]
    :return: 汉明距离
    """
    avg1, avg2 = np.mean(emb1, axis=-1), np.mean(emb2, axis=-1)
    binary1, binary2 = np.where(emb1 < avg1, 0, 1), np.where(emb2 < avg2, 0, 1)
    return len(np.nonzero(binary1 - binary2, )[0])


# 编辑距离