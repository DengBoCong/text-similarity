#! -*- coding: utf-8 -*-
""" 全局Base类
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class IdfBase(object):
    """ 实现有idf计算的基础类
    """

    def __init__(self, tokens_list: list = None, file_path: str = None, file_list: list = None, split: str = None):
        """ tokens_list、file_path、file_list三者传其一，tokens_list为文本列表时，split必传
        :param tokens_list: 已经分词的文本列表或token列表
        :param file_path: 已分词文本列表文件路径，一行一个文本
        :param file_list: 已分词的文本列表文件路径列表，一行一个文本
        :param split: 文本分隔符，list模式不传则每个element视为list，file模式必传
        :return: None
        """
        self.tokens_list = list()
        self.counts = list()
        self.document_count = 0
        self.token_docs = dict()
        self.length_average = 0.  # 文档平均长度

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
        """ 配合init初始化语料文本的相关特征变量
        :param tokens: tokens列表
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
        """ 配合init初始化语料文本的相关特征变量，file模式
        :param file_path: 文件路径
        :param split: 文本分隔符
        :return: None
        """
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().strip("\n")
                if line == "":
                    continue
                self._init_token_feature(tokens=line.split(split))
