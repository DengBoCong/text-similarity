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
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from typing import Any
from typing import NoReturn


class IdfBase(object):
    """ 实现了idf计算的基类
    """

    def __init__(self, tokens_list: list = None, file_path: str = None, file_list: list = None, split: str = None):
        """ tokens_list、file_path和file_list必传其一，当tokens_list时text list时，split必传
        :param tokens_list: text list或token list
        :param file_path: 已分词的文本数据路径，一行一个文本
        :param file_list: file list，每个文件一行一个文本
        :param split: 如果在list模式没有传，则视element为list，file模式必传
        :return: None
        """
        self.tokens_list = list()
        self.counts = list()
        self.document_count = 0
        self.token_docs = dict()
        self.length_average = 0.  # 平均文本长度

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

    def _init_token_feature(self, tokens: list) -> NoReturn:
        """ 初始化语料文本的特征变量
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

    def _init_file_token_feature(self, file_path: str, split: str) -> NoReturn:
        """ 初始化语料文本的特征变量，file模式
        :param file_path: file path
        :param split: 分隔符
        :return: None
        """
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().strip("\n")
                if line == "":
                    continue
                self._init_token_feature(tokens=line.split(split))


class LSH(abc.ABC):
    """ 实现了hash的LSH基类
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def search(self, *args, **kwargs):
        """ 匹配搜索，返回搜索结果
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    @staticmethod
    def hash(key: str, hash_obj: str = "md5") -> Any:
        """ 用于计算hash值
        :param key: hash key
        :param hash_obj: 用于计算hash值的方法
        :return:
        """
        if hash_obj == "md5":
            return hashlib.md5(key.encode("utf-8")).hexdigest()
        elif hash_obj == "sha1":
            return hashlib.sha1(key.encode("utf-8")).hexdigest()


class PcaBase(abc.ABC):
    """ 实现了PCA的基类
    """

    def __init__(self, svd_solver: str = "auto", component_type: str = "pca"):
        """
        :param svd_solver: svd solver
        :param component_type: component type
        :return: None
        """
        self.component = None
        self.svd_solver = svd_solver
        self.component_type = component_type

    def _get_component(self, n_components: int, component: Any = None) -> NoReturn:
        """ 获取PCA实现
        :param n_components: 输出数据的维度
        :param component: PCA的实现类
        :return: None
        """
        if component:
            if not hasattr(component, "fit") or not hasattr(component, "components_"):
                raise ValueError("The fit() method and components_ attributes must be implemented in the component")
            else:
                self.component = component
        elif self.component_type == "pca":
            self.component = PCA(n_components=n_components, svd_solver=self.svd_solver)
        elif self.component_type == "svd":
            self.component = TruncatedSVD(n_components=n_components, n_iter=7, random_state=0)
        else:
            raise ValueError("Please instantiate the PCA implementation class")
