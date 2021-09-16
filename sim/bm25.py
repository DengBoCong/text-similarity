#! -*- coding: utf-8 -*-
""" BM25及相关方法实现
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


class BM25(IdfBase):
    """ BM25
    """

    def __init__(self, tokens_list: list = None, file_path: str = None, file_list: list = None, split: str = None):
        """ tokens_list、file_path、file_list三者传其一，tokens_list为文本列表时，split必传
        :param tokens_list: 已经分词的文本列表或token列表
        :param file_path: 已分词文本列表文件路径，一行一个文本
        :param file_list: 已分词的文本列表文件路径列表，一行一个文本
        :param split: 文本分隔符，list模式不传则每个element视为list，file模式必传
        :return: None
        """
        super(BM25, self).__init__(tokens_list, file_path, file_list, split)

    def get_score(self, query: list, index: int, q_tf_dict: dict = None, q_total: int = 0,
                  if_tq: bool = True, e: int = 0.5, b=0.75, k1=2, k2=1.2) -> float:
        """ 计算文本序列与文本列表指定的文本序列的BM25相似度分数
        :param query: 文本序列
        :param index: 指定文本列表中的文本序列索引
        :param q_tf_dict: query的token字典，用来配合批量计算分数使用，提高效率
        :param q_total: query的token总数，同上
        :param if_tq: 是否刻画单词与query之间的相关性，长的query默认开启
        :param e: 调教系数
        :param b: 可调参数，(0,1)
        :param k1: 可调正参数，[1.2, 2.0]
        :param k2: 可调正参数，[1.2, 2.0]
        :return: BM25相似分数
        """
        score = 0.
        d_total = sum(self.counts[index].values())

        if if_tq and not q_tf_dict:
            # 计算query词数
            q_total = len(query)
            q_tf_dict = dict()
            for token in query:
                q_tf_dict[token] = q_tf_dict.get(token, 0) + 1

        for token in query:
            if token not in self.counts[index]:
                continue

            score += self._cal_bm25_value(token=token, count=self.counts[index], total=d_total, e=e, b=b,
                                          k1=k1, k2=k2, if_tq=if_tq, q_tf_dict=q_tf_dict, q_total=q_total)

        return score

    def get_score_list(self, query: list, top_k: int = 0, if_tq: bool = True,
                       e: int = 0.5, b=0.75, k1=2, k2=1.2) -> list:
        """ 检索文本列表中BM25分数最高的前top-k个文本序列，当
            top-k为0时，返回文本列表中所有文本序列与指定文本序列
            的BM25分数
        :param query: 文本序列
        :param top_k: 返回的数量
        :param if_tq: 是否刻画单词与query之间的相关性，长的query默认开启
        :param e: 调教系数
        :param b: 可调参数，(0,1)
        :param k1: 可调正参数，[1.2, 2.0]
        :param k2: 可调正参数，[1.2, 2.0]
        :return: BM25分数列表
        """
        scores = list()
        q_tf_dict = None
        q_total = 0

        if if_tq:
            # 计算query词数
            q_total = len(query)
            q_tf_dict = dict()
            for token in query:
                q_tf_dict[token] = q_tf_dict.get(token, 0) + 1

        for i in range(self.document_count):
            node = (i, self.get_score(query=query, index=i, q_tf_dict=q_tf_dict,
                                      q_total=q_total, if_tq=if_tq, e=e, b=b, k1=k1, k2=k2))
            scores.append(node)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores if top_k == 0 else scores[:top_k]

    def weight(self, e: int = 0.5, b=0.75, k1=2, k2=1.2, pad_size: int = None, padding: str = "pre",
               if_tq: bool = False, truncating: str = "pre", value: float = 0., d_type: str = "float32") -> Any:
        """ 对tokens_list中的句向量进行 BM25加权转换，当
            不传pad_size时，则返回对应的权重list，当传入
            pad_size时(对其他参数生效)，则返回补齐的numpy的矩阵张量
        :param e: 调教系数
        :param b: 可调参数，(0,1)
        :param k1: 可调正参数，[1.2, 2.0]
        :param k2: 可调正参数，[1.2, 2.0]
        :param pad_size: 填充的最大长度
        :param padding: 填充类型，pre在前，post在后
        :param if_tq: 是否刻画单词与query之间的相关性，长的query默认开启
        :param truncating: 截断类型，pre在前，post在后
        :param value: 填充值值
        :param d_type: 输出类型
        :return: tokens_list的BM25权重
        """
        if not self.tokens_list:
            raise ValueError("tokens_list is not initialized in th init func")

        if not pad_size:
            result_list = list()
            for tokens, count in zip(self.tokens_list, self.counts):
                node_list = list()
                total = len(tokens)
                for token in tokens:
                    node_list.append(self._cal_bm25_value(token=token, count=count, total=total,
                                                          e=e, b=b, k1=k1, k2=k2, if_tq=if_tq))

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
                    trunc.append(self._cal_bm25_value(token=token, count=count, total=total,
                                                      e=e, b=b, k1=k1, k2=k2, if_tq=if_tq))

                trunc = np.asarray(trunc, dtype=d_type)
                if padding == "post":
                    result[i, :len(trunc)] = trunc
                elif padding == "pre":
                    result[i, -len(trunc):] = trunc
                else:
                    raise ValueError('Padding type "%s" not understood' % padding)

            return result

    def _cal_bm25_value(self, token: str, count: dict, total: int, e: int = 0.5, b=0.75, k1=2,
                        k2=1.2, if_tq: bool = True, q_tf_dict: dict = None, q_total: int = 0) -> float:
        """ 内部计算bm25值
        :param token: 当前计算的token
        :param count: 当前计算的列表频次字典
        :param total: 当前计算的列表token总数
        :param e: 调教系数
        :param b: 可调参数，(0,1)
        :param k1: 可调正参数，[1.2, 2.0]
        :param k2: 可调正参数，[1.2, 2.0]
        :param if_tq: 是否刻画单词与query之间的相关性，长的query默认开启
        :param q_tf_dict: query的token字典
        :param q_total: query的token总数
        :return: bm25值
        """
        idf = math.log((self.document_count + e) / (self.token_docs[token] + e))
        tf_td = count[token] / total
        sim_td = ((k1 + 1) * tf_td) / (k1 * (1 - b + b * total / self.length_average) + tf_td)

        sim_tq = 1.
        if if_tq:
            tf_tq = tf_td if not q_tf_dict else q_tf_dict[token] / q_total
            sim_tq = ((k2 + 1) * tf_tq) / (k2 + tf_tq)

        return idf * sim_td * sim_tq

    def extract_keywords(self, query: list = None, query_file_path: str = None,
                         idf_dict: dict = None, idf_file: str = None):
        """ 提取关键词
        """
        pass
