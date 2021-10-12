#! -*- coding: utf-8 -*-
""" Implementation of SIF and uSIF
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from sim.base import PcaBase
from typing import Any


class SIF(PcaBase):
    """ Smooth Inverse Frequency (SIF)
    Calculate the weighted average value for the token inside the
    sentence, and subtract the projection of all word vectors on
    the first principal component, and then get sentence embedding

    Example:
        from sentence2vec.transform import SIF
        sif = SIF(n_components=5, component_type="svd")
        sif.fit(tokens_list=sentences, vector_list=vector)

    PCA calculation depend on implementation of PAC and TruncatedSVD in
    scikit-learn, custom implementation can also be passed in
    """

    def __init__(self, n_components: int, parameter: float = 1e-3, word_freq: dict = None,
                 svd_solver: str = "auto", component_type: str = "pca", name: str = None):
        """
        :param n_components: desired dimensionality of output data
        :param parameter: adjustable parameter
        :param word_freq: word freq dict
        :param svd_solver: svd solver
        :param component_type: component type
        :param name:
        :return: None
        """
        super(SIF, self).__init__(svd_solver=svd_solver, component_type=component_type)
        self.n_components = n_components
        self.parameter = parameter
        self.word_freq = word_freq
        self.name = name
        self.pairs = None
        self.prob_weight = dict()
        self.n_samples = None

    def fit(self, tokens_list: list, vector_list: list, component: Any = None) -> None:
        """ Construct word vector

        :param tokens_list: the token list of the original sentence, shape = [counts, seq_len]
        :param vector_list: the token embedding, shape = [counts, seq_len, feature]
        :param component: calculating PCA implementation class
        :return: None
        """
        if self.word_freq and isinstance(self.word_freq, (dict, collections.Counter)):
            raise TypeError("word_freq must be dict")
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
        self._get_component(self.n_components, component)

    def _get_words_weight(self, words: list) -> list:
        """ get the sentences word freq weight

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

    def transform(self, n_features: int) -> np.ndarray:
        """ Conversion word vector

        :param n_features: feature size
        :return: vector
        """
        sentence_list = np.zeros((self.n_samples, n_features))
        for index, (tokens, vector) in enumerate(self.pairs):
            sentence_list[index, :] = np.dot(self._get_words_weight(tokens), vector) / len(tokens)

        self.component.fit(sentence_list)
        u = self.component.components_

        return sentence_list - sentence_list.dot(u.transpose()).dot(u)


class uSIF(PcaBase):
    """ unsupervised Smooth Inverse Frequency (uSIF)

    Normalize the word vector, then use their weighted average to
    calculate sentence vectors. And subtract the projections on the
    first m principal components, and then get the sentence embedding

    Example:
        from sentence2vec.transform import uSIF
        usif = uSIF(n_components=5, n=1, component_type="svd")
        usif.fit(tokens_list=sentences, vector_list=vector)

    PCA calculation depend on implementation of PAC and TruncatedSVD in
    scikit-learn, custom implementation can also be passed in
    """

    def __init__(self, n_components, n=11, word_freq=None,
                 svd_solver="auto", component_type="pca", name=None):
        """
        :param n_components: desired dimensionality of output data
        :param n: adjustable parameter
        :param word_freq: word freq dict
        :param svd_solver: svd solver
        :param component_type: component type
        :param name:
        :return: None
        """
        super(uSIF, self).__init__(svd_solver=svd_solver, component_type=component_type)
        self.n_components = n_components
        self.n = n
        self.word_freq = word_freq
        self.name = name
        self.pairs = None
        self.parameter = None
        self.prob_weight = None
        self.n_samples = None

    def fit(self, tokens_list: list, vector_list: list, component: Any = None) -> None:
        """ Construct word vector

        :param tokens_list: the token list of the original sentence, shape = [counts, seq_len]
        :param vector_list: the token embedding, shape = [counts, seq_len, feature]
        :param component: calculating PCA implementation class
        :return: None
        """
        if not (isinstance(self.n, int) and self.n > 0):
            raise TypeError("n must be a positive integer")

        if self.word_freq and isinstance(self.word_freq, (dict, collections.Counter)):
            raise TypeError("word_freq must be dict")
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
            raise ValueError("n is too large, please reset")

        self.parameter = (1 - alpha) / (alpha * z)
        self.prob_weight = lambda word: (self.parameter / (0.5 * self.parameter + self.word_freq[word] / total_word))
        self.n_samples = len(tokens_list)
        self.pairs = zip(tokens_list, vector_list)
        self._get_component(self.n_components, component)

    def transform(self, n_features: int) -> np.ndarray:
        """ Conversion word vector

        :param n_features: feature size
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
