#! -*- coding: utf-8 -*-
""" Implementation of the TF-IDF
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from sim.base import IdfBase
from typing import Any


class TFIdf(IdfBase):
    """ TF-IDF
    """

    def __init__(self, tokens_list: list = None, file_path: str = None, file_list: list = None, split: str = None):
        """ tokens_list、file_path、file_list三者传其一，tokens_list为文本列表时，split必传
        :param tokens_list: 已经分词的文本列表或token列表
        :param file_path: 已分词文本列表文件路径，一行一个文本
        :param file_list: 已分词的文本列表文件路径列表，一行一个文本
        :param split: 文本分隔符，list模式不传则每个element视为list，file模式必传
        """
        super(TFIdf, self).__init__(tokens_list, file_path, file_list, split)

    def get_score(self, query: list, index: int, e: int = 0.5) -> float:
        """ 计算查询语句与语料库中指定文本的tf-idf相似分数
        :param query: 查询文本
        :param index: 语料库中文本索引
        :param e: 调教系数
        :return: tf-idf相似分数
        """
        score = 0.0
        total = sum(self.counts[index].values())
        for token in query:
            if token not in self.counts[index]:
                continue
            idf = math.log((self.document_count + e) / (self.token_docs[token] + e))
            score += (self.counts[index][token] / total) * idf

        return score

    def get_score_list(self, query: list, top_k: int = 0, e: int = 0.5):
        """ 检索文本列表中tf-idf分数最高的前top-k个文本序列，当top-k为
            0时，返回文本列表中所有文本序列与指定文本序列的td-idf分数
        :param query: 文本序列
        :param top_k: 返回的数量
        :param e: 调教系数
        :return: tf-idf分数列表，列表中的element为(index, score)
        """
        scores = list()
        for i in range(self.document_count):
            node = (i, self.get_score(query=query, index=i, e=e))
            scores.append(node)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores if top_k == 0 else scores[:top_k]

    def weight(self, e: int = 0.5, pad_size: int = None, padding: str = "pre",
               truncating: str = "pre", value: float = 0., d_type: str = "float32") -> Any:
        """ 对tokens_list中的句向量进行TF-IDF加权转换，当
            不传pad_size时，则返回对应的权重list，当传入
            pad_size时(对其他参数生效)，则返回补齐的numpy的矩阵张量
        :param e: 调教系数
        :param pad_size: 填充的最大长度
        :param padding: 填充类型，pre在前，post在后
        :param truncating: 截断类型，pre在前，post在后
        :param value: 填充值值
        :param d_type: 输出类型
        :return: tokens_list的TF-IDF权重
        """
        if not self.tokens_list:
            raise ValueError("tokens_list is not initialized in th init func")

        if not pad_size:
            result_list = list()
            for tokens, count in zip(self.tokens_list, self.counts):
                node_list = list()
                total = len(tokens)
                for token in tokens:
                    idf = math.log((self.document_count + e) / (self.token_docs[token] + e))
                    node_list.append((count[token] / total) * idf)

                result_list.append(node_list)
            return result_list
        else:
            is_d_type_str = np.issubdtype(d_type, np.str_) or np.issubdtype(d_type, np.unicode_)
            if isinstance(value, str) and d_type != object and not is_d_type_str:
                raise ValueError("`d_type` {} is not compatible with `value`'s type: {}\n"
                                 "You should set `d_type=object` for variable length strings."
                                 .format(d_type, type(value)))

            result = np.full(shape=(self.document_count, pad_size), fill_value=value, dtype=d_type)
            for i, (tokens, count) in enumerate(zip(self.tokens_list, self.counts)):
                total = len(tokens)

                if truncating == "pre":
                    tokens = tokens[-pad_size:]
                elif truncating == "post":
                    tokens = tokens[:pad_size]
                else:
                    raise ValueError('Truncating type "%s" '
                                     'not understood' % truncating)

                trunc = list()
                for token in tokens:
                    idf = math.log((self.document_count + e) / (self.token_docs[token] + e))
                    trunc.append(count[token] / total * idf)

                trunc = np.asarray(trunc, dtype=d_type)
                if padding == "post":
                    result[i, :len(trunc)] = trunc
                elif padding == "pre":
                    result[i, -len(trunc):] = trunc
                else:
                    raise ValueError('Padding type "%s" not understood' % padding)

            return result

    def extract_keywords(self, query: list = None, query_file_path: str = None,
                         idf_dict: dict = None, idf_file: str = None):
        """ 提取关键词
        """
        pass
