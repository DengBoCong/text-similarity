#! -*- coding: utf-8 -*-
""" Sentence Embedding transform
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def weighted_average_sim(emb1, emb2):
    """ 计算句子分数

    :param emb1:
    :param emb2:
    :return:
    """
    inn = (emb1 * emb2).sum(axis=1)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
    scores = inn / emb1norm / emb2norm
    return scores
