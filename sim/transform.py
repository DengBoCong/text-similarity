#! -*- coding: utf-8 -*-
""" Sentence Embedding transform
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
import collections
import numpy as np
from sim.tools import counter
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


class Base(abc.ABC):
    def __init__(self, svd_solver="auto", component_type="pca", **kwargs):
        super().__init__()

        self.component = None
        self.svd_solver = svd_solver
        self.component_type = component_type

    def _get_component(self, n_components, component=None, **kwargs):
        """ 获取实现类

        :param component: 计算主成分实现类
        :param kwargs:
        :return: None
        """
        if component:
            if not hasattr(component, "fit") or not hasattr(component, "components_"):
                raise ValueError("component实现中必须实现fit()方法、components_属性")
            else:
                self.component = component
        elif self.component_type == "pca":
            self.component = PCA(n_components=n_components, svd_solver=self.svd_solver)
        elif self.component_type == "svd":
            self.component = TruncatedSVD(n_components=n_components, n_iter=7, random_state=0)
        else:
            raise ValueError("请实例化主成分实现类")


class SIF(Base):
    """ Smooth Inverse Frequency (SIF)

    对句子内部token表示计算加权平均值，并减去所有词向量
    在第一个主成分上的投影，进而得到Sentence Embedding

    Example:
        from sentence2vec.transform import SIF
        sif = SIF(n_components=5, component_type="svd")
        sif.fit(tokens_list=sentences, vector_list=vector)

    主成分计算依赖scikit-learn中PAC和TruncatedSVD实现，也可传入自定义实现
    """

    def __init__(self, n_components, parameter=1e-3, word_freq=None,
                 svd_solver="auto", component_type="pca", name=None, **kwargs):
        super().__init__(svd_solver=svd_solver, component_type=component_type, **kwargs)
        self.n_components = n_components
        self.parameter = parameter
        self.word_freq = word_freq
        self.name = name
        self.pairs = None
        self.prob_weight = dict()
        self.n_samples = None

    def fit(self, tokens_list, vector_list, component=None, **kwargs):
        """词向量数据构建

        :param tokens_list: 原句子的token列表，shape = [counts, seq_len]
        :param vector_list: 句子的token向量化列表，shape = [counts, seq_len, feature]
        :param component: 计算主成分实现类
        :return:
        """
        if self.word_freq and isinstance(self.word_freq, (dict, collections.Counter)):
            raise TypeError("word_freq必须为词频字典")
        else:
            self.word_freq = collections.Counter()
            for tokens in tokens_list:
                for token in tokens:
                    self.word_freq[token] += 1

        total_word = sum(self.word_freq.values())
        for key, value in self.word_freq.items():
            self.prob_weight[key] = self.parameter / (self.parameter + value / total_word)

        self.n_samples = len(tokens_list)
        self.pairs = zip(tokens_list, vector_list)
        self._get_component(self.n_components, component, **kwargs)

    def _get_words_weight(self, words):
        """ 获取sentences词频权重

        :param words: sentences
        :return: count
        """
        weights = []
        for word in words:
            if word in self.prob_weight:
                weights.append(self.prob_weight[word])
            else:
                weights.append(1.0)

        return weights

    def transform(self, n_features):
        """ 词向量转换

        :param n_features: 特征维大小
        :return: vector
        """
        sentence_list = np.zeros((self.n_samples, n_features))
        for index, (tokens, vector) in enumerate(self.pairs):
            sentence_list[index, :] = np.dot(self._get_words_weight(tokens), vector) / len(tokens)

        self.component.fit(sentence_list)
        u = self.component.components_

        return sentence_list - sentence_list.dot(u.transpose()).dot(u)


class uSIF(Base):
    """ unsupervised Smooth Inverse Frequency (uSIF)

    对句子的词向量进行归一化，然后使用它们的加权平均计算句向
    量，并减去前m个主成分上的投影，进而得到Sentence Embedding

    Example:
        from sentence2vec.transform import uSIF
        usif = uSIF(n_components=5, n=1, component_type="svd")
        usif.fit(tokens_list=sentences, vector_list=vector)

    主成分计算依赖scikit-learn中PAC和TruncatedSVD实现，也可传入自定义实现
    """

    def __init__(self, n_components, n=11, word_freq=None,
                 svd_solver="auto", component_type="pca", name=None, **kwargs):
        super().__init__(svd_solver=svd_solver, component_type=component_type, **kwargs)
        self.n_components = n_components
        self.n = n
        self.word_freq = word_freq
        self.name = name
        self.pairs = None
        self.parameter = None
        self.prob_weight = None
        self.n_samples = None

    def fit(self, tokens_list, vector_list, component=None, **kwargs):
        """词向量数据构建

        :param tokens_list: 原句子的token列表，shape = [counts, seq_len]
        :param vector_list: 句子的token向量化列表，shape = [counts, seq_len, feature]
        :param component: 计算主成分实现类
        :return:
        """
        if not (isinstance(self.n, int) and self.n > 0):
            raise TypeError("n必须为正整数")

        if self.word_freq and isinstance(self.word_freq, (dict, collections.Counter)):
            raise TypeError("word_freq必须为词频字典")
        else:
            self.word_freq = collections.Counter()
            for tokens in tokens_list:
                for token in tokens:
                    self.word_freq[token] += 1

        total_word = sum(self.word_freq.values())
        vocab_size = float(len(self.word_freq.keys()))
        threshold = 1 - (1 - 1 / vocab_size) ** self.n
        alpha = len([w for w in self.word_freq.keys() if (self.word_freq[w] / total_word) > threshold]) / vocab_size
        z = 0.5 * vocab_size

        if alpha == 0.0:
            raise ValueError("n设置过大，请重新设置")

        self.parameter = (1 - alpha) / (alpha * z)
        self.prob_weight = lambda word: (self.parameter / (0.5 * self.parameter + self.word_freq[word] / total_word))
        self.n_samples = len(tokens_list)
        self.pairs = zip(tokens_list, vector_list)
        self._get_component(self.n_components, component, **kwargs)

    def transform(self, n_features):
        """ 词向量转换

        :param n_features: 特征维大小
        :return: vector
        """
        proj = lambda a, b: a.dot(b.transpose()) * b
        sentence_list = np.zeros((self.n_samples, n_features))

        for index, (tokens, vector) in enumerate(self.pairs):
            vector_t = vector * (1.0 / np.linalg.norm(vector, axis=0))
            vector_t = np.array([self.prob_weight(t) * vector_t[i, :] for i, t in enumerate(tokens)])
            sentence_list[index, :] = np.mean(vector_t, axis=0)

        self.component.fit(sentence_list)
        for i in range(self.n_components):
            lambda_i = (self.component.singular_values_[i] ** 2) / (self.component.singular_values_ ** 2).sum()
            pc = self.component.components_[i]
            sentence_list = [vs - lambda_i * proj(vs, pc) for vs in sentence_list]

        return np.array(sentence_list)





class BM25(object):
    def __init__(self, b=0.75, k1=2, k2=1, e=0.5):
        """
        :param b:
        :param k1: 范围[1.2, 2.0]
        :param k2:
        :param e: IDF计算调教系数
        """
        self.b = b
        self.k1 = k1
        self.k2 = k2
        self.e = e

    def get_score(self, index, query):
        """ 计算查询语句与语料库中指定文本的bm25相似分数

        :param index: 语料库中文本索引
        :param query: 查询文本
        :return: bm25相似分数
        """
        pass

    def bm25_weight(self, tokens_list, pad_size=None, counts=None, d_type=np.float):
        """ 计算token列表的BM25权重

        :param tokens_list: 已经分词的token列表，shape = [counts, seq_len]，seq_len可以不等长
        :param pad_size: seq_len填充大小，默认不进行填充（填充0值）
        :param counts: 词频次列表
        :param d_type: 数据类型
        :return: idf列表，pad_size为空则为list，不为空则为np.array
        """
        if counts is None:
            counts = counter(tokens_list)

        idf_dict = dict()
        token_total = len(tokens_list)
        avg_doc_len = sum([len(tokens) for tokens in tokens_list]) / token_total

        if not pad_size:
            tokens_weight = list()
            for index, tokens in enumerate(tokens_list):
                token_weight = list()
                doc_len = len(counts[index])
                for token in tokens:
                    if not idf_dict.get(token):
                        total = sum(1 for count in counts if count.get(token))
                        idf_dict[token] = math.log((token_total - total + self.e) / (total + self.e) + 1)
                    freq = counts[index][token]
                    weight = idf_dict[token] * (freq * (self.k1 + 1) / (freq + self.k1 * (
                            1 - self.b + self.b * doc_len / avg_doc_len))) * (freq * (self.k2 + 1) / (freq + self.k2))
                    token_weight.append(weight)
                tokens_weight.append(token_weight)
        else:
            tokens_weight = np.zeros(shape=(len(tokens_list), pad_size), dtype=d_type)
            for row, tokens in enumerate(tokens_list):
                doc_len = len(counts[row])
                for col, token in enumerate(tokens):
                    if not idf_dict.get(token):
                        total = sum(1 for count in counts if count.get(token))
                        idf_dict[token] = math.log((token_total - total + self.e) / (total + self.e) + 1)
                    freq = counts[row][token]
                    tokens_weight[row, col] = idf_dict[token] * (freq * (self.k1 + 1) / (freq + self.k1 * (
                            1 - self.b + self.b * doc_len / avg_doc_len))) * (freq * (self.k2 + 1) / (freq + self.k2))

        return idf_dict, tokens_weight

    def transform(self, tokens_list, vector_list, mask, d_type=np.float):
        """
        :param tokens_list: 原句子的token列表，shape = [counts, seq_len]
        :param vector_list: 句子的token向量化列表，shape = [counts, seq_len, feature]，seq_len严格等长
        :param mask: tokens填充mask，非填充为1，填充为0
        :param d_type: 数据类型
        """
        vector_list = np.array(vector_list, dtype=d_type)
        weight = self.bm25_weight(tokens_list, vector_list.shape[1], None, d_type)
        result = vector_list * weight[:, :, np.newaxis] * mask[:, :, np.newaxis]
        return np.mean(result, axis=1)


class WMD(Base):
    pass
