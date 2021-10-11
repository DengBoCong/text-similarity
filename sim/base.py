#! -*- coding: utf-8 -*-
""" Global Base Class
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import hashlib
from typing import Any


class IdfBase(object):
    """ Implement the base class with idf cal
    """

    def __init__(self, tokens_list: list = None, file_path: str = None, file_list: list = None, split: str = None):
        """ tokens_list、file_path and file_list must pass one, when tokens_list is a text list, split must pass
        :param tokens_list: text list or token list
        :param file_path: the path of the word segmented text file, one text per line
        :param file_list: file list, one text per line
        :param split: if the list mode is not passed, element is regarded as a list, and the file mode must be passed
        :return: None
        """
        self.tokens_list = list()
        self.counts = list()
        self.document_count = 0
        self.token_docs = dict()
        self.length_average = 0.  # Average doc length

        if tokens_list and not split:
            for tokens in tokens_list:
                self._init_token_feature(tokens=tokens)
        elif tokens_list and split:
            for tokens in tokens_list:
                self._init_token_feature(tokens=tokens.split(split))
        elif file_path:
            if not split:
                raise ValueError("In file mode, split must be transmitted")
            self._init_file_token_feature(file_path=file_path, split=split)
        elif file_list:
            if not split:
                raise ValueError("In file list mode, split must be transmitted")
            for file in file_list:
                self._init_file_token_feature(file_path=file, split=split)

        self.length_average /= self.document_count

    def _init_token_feature(self, tokens: list) -> None:
        """ Initialize the feature variables of the corpus text
        :param tokens: tokens list
        :return: None
        """
        self.tokens_list.append(tokens)
        self.document_count += 1
        self.length_average += len(tokens)

        td_tf_dict = dict()
        for token in tokens:
            td_tf_dict[token] = td_tf_dict.get(token, 0) + 1

        self.counts.append(td_tf_dict)

        for token in set(tokens):
            self.token_docs[token] = self.token_docs.get(token, 0) + 1

    def _init_file_token_feature(self, file_path: str, split: str) -> None:
        """ Initialize the feature variables of the corpus text, file mode
        :param file_path: file path
        :param split:
        :return: None
        """
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().strip("\n")
                if line == "":
                    continue
                self._init_token_feature(tokens=line.split(split))


class LSH(abc.ABC):
    """ Base class that implements LSH
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def search(self, *args, **kwargs):
        """ Match search, return search result
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    @staticmethod
    def hash(key: str, hash_obj: str = "md5") -> Any:
        """ Used to calculate the hash value
        :param key: hash key
        :param hash_obj: the method used to calculate the hash
        :return:
        """
        if hash_obj == "md5":
            return hashlib.md5(key.encode("utf-8")).hexdigest()
        elif hash_obj == "sha1":
            return hashlib.sha1(key.encode("utf-8")).hexdigest()


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
