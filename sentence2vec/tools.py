#! -*- coding: utf-8 -*-
""" Coding Tools
"""

# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

import math


def _counter(sentences):
    """ 计算分词句子列表的词频次

    :param sentences: 分词句子列表
    :return 词频次列表
    """
    word_counts = []
    for sentence in sentences:
        count = {}
        for word in sentence:
            if not count.get(word):
                count.update({word: 1})
            elif count.get(word):
                count[word] += 1
        word_counts.append(count)
    return word_counts


def tf(sentences, counts=None):
    """ 计算分词句子列表每个词的tf

    :param sentences: 分词句子列表
    :param counts: 词频次列表
    :return: tf列表
    """
    if counts is None:
        counts = _counter(sentences)
    tfs = list()
    for count in counts:
        tf_dict, total = dict(), sum(count.values())
        for key in count.keys():
            if not tf_dict.get(key):
                tf_dict[key] = count[key] / total
        tfs.append(tf_dict)

    return tfs


def idf(sentences, counts=None):
    """ 计算分词句子列表每个词的idf

    :param sentences: 分词句子列表
    :param counts: 词频次列表
    :return: tf列表
    """
    if counts is None:
        counts = _counter(sentences)
    idf_dict = dict()
    sentence_total = len(sentences)
    for sentence in sentences:
        for word in sentence:
            if not idf_dict.get(word):
                total = sum(1 for count in counts if count.get(word))
                idf_dict[word] = math.log(sentence_total / (total + 1))

    return idf_dict


def tf_idf(sentences, counts=None):
    """ 计算分词句子列表每个词的TF-IDF

    :param sentences: 分词句子列表
    :param counts: 词频次列表
    :return: TF-IDF列表
    """
    if counts is None:
        counts = _counter(sentences)
    idf_dict = idf(sentences, counts)
    tfs = tf(sentences, counts)
    for tf_dict in tfs:
        for key in tf_dict.keys():
            tf_dict[key] *= idf_dict[key]
    return tfs
