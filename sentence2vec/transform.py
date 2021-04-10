#! -*- coding: utf-8 -*-

import abc
import collections
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


class Base(abc.ABC):
    def __init__(self, ):
        super().__init__()


class SIF(Base):
    """ Smooth Inverse Frequency (SIF)

    对句子内部token表示计算加权平均值，并减去所有词向量
    在第一个主成分上的投影，进而得到Sentence Embedding

    主成分计算依赖scikit-learn中PAC和TruncatedSVD实现，也可传入自定义实现
    """

    def __init__(self, embedding_size, parameter=1e-3, word_freq=None,
                 component_type="pca", name=None, **kwargs):
        super().__init__()
        self.embedding_size = embedding_size
        self.parameter = parameter
        self.word_freq = word_freq
        self.name = name
        self.pairs = None
        self.component = None
        self.component_type = component_type

    def build(self, tokens_list, vector_list, component=None, **kwargs):
        """词向量数据构建

        :param tokens_list: 原句子的token列表
        :param vector_list: 句子的token向量化列表
        :param component: 计算主成分实现类
        :return:
        """
        if self.word_freq and isinstance(self.word_freq, (dict, collections.Counter)):
            raise ValueError("word_freq必须为词频字典")
        else:
            self.word_freq = collections.Counter()
            for tokens in tokens_list:
                for token in tokens:
                    self.word_freq[token] += 1

        self.pairs = zip(tokens_list, vector_list)
        self._get_component(component, **kwargs)

    def _get_component(self, component=None, **kwargs):
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
            self.component = PCA(n_components=self.embedding_size)
        elif self.component_type == "svd":
            self.component = TruncatedSVD(n_components=self.embedding_size, n_iter=7, random_state=0)
        else:
            raise ValueError("请实例化主成分实现类")

    def _get_word_freq(self, word):
        """ 获取word词频，不存在只计数1

        :param word: text word
        :return: count
        """
        if word in self.word_freq:
            return self.word_freq[word]
        else:
            return 1.0

    def transform(self):
        sentence_list = []
        for tokens, vector in self.pairs:
            vs = np.zeros(self.embedding_size)
            sentence_length = len(tokens)
            for token in tokens:
                a_value = self.parameter / (self.parameter + self._get_word_freq(token))
                vs = np.add(vs, np.multiply(a_value, ))
