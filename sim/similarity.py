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
    """ Calculate the Euclidean distance between feature vectors
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: dimension of effect
    :return: Euclidean distance
    """
    return np.sqrt(np.square(emb1 - emb2).sum(axis=axis))


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray, dist: bool = False, axis: int = -1) -> Any:
    """ calculate the cosine similarity score between feature vectors
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param dist: whether to return the cosine distance (after normalization), [0, 1]
    :param axis: dimension of effect
    :return: Cosine similarity
    """
    mod = np.linalg.norm(emb1, axis=axis) * np.linalg.norm(emb2, axis=axis)
    if np.all(mod == 0):
        raise RuntimeError("cosine similarity divisor is zero")
    cos = np.sum(emb1 * emb2, axis=axis, dtype=float) / mod

    return (1 - cos) / 2.0 if dist else cos


def manhattan_dist(emb1: np.ndarray, emb2: np.ndarray, axis: int = -1) -> Any:
    """ calculate the Manhattan distance between feature vectors
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: dimension of effect
    :return: Manhattan distance
    """
    return np.sum(np.abs(emb1 - emb2), axis=axis)


def minkowsk_dist(emb1: np.ndarray, emb2: np.ndarray, p: int, axis: int = -1) -> Any:
    """ calculate the Minkowsk distance between feature vectors
    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param p: Norm
    :param axis: dimension of effect
    :return: Minkowsk distance
    """
    tmp = np.sum(np.power(emb1 - emb2, p), axis=axis)
    return np.power(tmp, 1 / p)


def hamming_dist(emb1: np.ndarray, emb2: np.ndarray, axis: int = -1) -> np.ndarray:
    """ calculate the Hamming distance between feature vectors

    :param emb1: shape = [feature,]
    :param emb2: shape = [feature,]
    :param axis: dimension of effect
    :return: Hamming distance
    """
    if len(emb1.shape) > 1 or len(emb2.shape) > 1:
        raise RuntimeError("the shape of emb1 and emb2 must be [feature,]")

    avg1, avg2 = np.mean(emb1, axis=axis), np.mean(emb2, axis=axis)
    binary1, binary2 = np.where(emb1 < avg1, 0, 1), np.where(emb2 < avg2, 0, 1)
    return len(np.nonzero(binary1 - binary2)[0])


def jaccard_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """ calculate the Jaccard similarity between feature vectors

    :param emb1: shape = [feature,]
    :param emb2: shape = [feature,]
    :return: Jaccard similarity
    """
    up = np.double(np.bitwise_and((emb1 != emb2), np.bitwise_or(emb1 != 0, emb2 != 0)).sum())
    down = np.double(np.bitwise_or(emb1 != 0, emb2 != 0).sum())
    d1 = (up / down)
    return d1


def pearson_similarity(emb1: np.ndarray, emb2: np.ndarray, axis: int = -1) -> np.ndarray:
    """ calculate the Pearson similarity between feature vectors

    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: dimension of effect
    :return: Pearson similarity
    """
    diff1, diff2 = emb1 - np.mean(emb1, axis=axis)[..., np.newaxis], emb2 - np.mean(emb2, axis=axis)[..., np.newaxis]
    up = np.sum(diff1 * diff2, axis=axis)
    down = np.sqrt(np.sum(np.square(diff1), axis=axis)) * np.sqrt(np.sum(np.square(diff2), axis=axis))
    return np.divide(up, down)


def mahalanobis_dist(emb1: np.ndarray, emb2: np.ndarray) -> list:
    """ calculate the Mahalanobis distance between feature vectors
    :param emb1: shape = [feature,]
    :param emb2: shape = [feature,]
    :return: Mahalanobis distance
    """
    x = np.vstack([emb1, emb2])
    xt = x.T
    si = np.linalg.inv(np.cov(x))
    n = xt.shape[0]
    d1 = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = xt[i] - xt[j]
            d = np.sqrt(np.dot(np.dot(delta, si), delta.T))
            d1.append(d)

    return d1


def kl_divergence(emb1: np.ndarray, emb2: np.ndarray, axis: int = -1) -> np.ndarray:
    """ calculate the KL divergence between feature vectors

    :param emb1: shape = [..., feature]
    :param emb2: shape = [..., feature]
    :param axis: dimension of effect
    :return: KL divergence
    """
    return np.sum(emb1 * np.log(np.divide(emb1, emb2)), axis=axis)


def levenshtein_dist(str1: str, str2: str):
    """ Calculate the edit distance between two strings

    :param str1: string
    :param str2: string
    :return: Edit distance
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]
