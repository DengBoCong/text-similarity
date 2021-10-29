#! -*- coding: utf-8 -*-
""" Implementation of BM25
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
        """ tokens_list、file_path和file_list必传其一，当tokens_list时text list时，split必传
        :param tokens_list: text list或token list
        :param file_path: 已分词的文本数据路径，一行一个文本
        :param file_list: file list，每个文件一行一个文本
        :param split: 如果在list模式没有传，则视element为list，file模式必传
        """
        super(BM25, self).__init__(tokens_list, file_path, file_list, split)

    def get_score(self, query: list, index: int, q_tf_dict: dict = None, q_total: int = 0,
                  if_tq: bool = True, e: int = 0.5, b=0.75, k1=2, k2=1.2) -> float:
        """ 计算query与指定index的seq之间的BM25分数
        :param query: token list
        :param index: 指定seq的index
        :param q_tf_dict: token dict
        :param q_total: query的token数量
        :param if_tq: 是否计算word和query之间的相关性，长query建议开启
        :param e: 可调整参数
        :param b: 可调整参数，(0,1)
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :return: BM25分数
        """
        score = 0.
        d_total = sum(self.counts[index].values())

        if if_tq and not q_tf_dict:
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
        """ 检索列表中与query的BM25分数最高的top_k个seq。当top_k为0时，返回所有的BM25分数，按照降序排列
        :param query: token list
        :param top_k: 返回结果数量
        :param if_tq: 是否计算word和query之间的相关性，长query建议开启
        :param e: 可调整参数
        :param b: 可调整参数，(0,1)
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :return: BM25分数列表
        """
        scores = list()
        q_tf_dict = None
        q_total = 0

        if if_tq:
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
        """ 对tokens_list中的句子向量进行BM25加权转换。当未传pad_size时，
            返回转换weight list。当传入pad_size时，返回numpy tensor
        :param e: 可调整参数
        :param b: 可调整参数，(0,1)
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :param pad_size: 填充到大小
        :param padding: 填充类型，pre/post
        :param if_tq: 是否计算word和query之间的相关性，长query建议开启
        :param truncating: 截断类型，pre/post
        :param value: 填充值
        :param d_type: 类型
        :return: BM25权重矩阵
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
        """ 计算BM25值
        :param token: 当前计算的token
        :param count: 词频词典
        :param total: tokens总数
        :param e: 可调整参数
        :param b: 可调整参数，(0,1)
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :param if_tq: 是否计算word和query之间的相关性，长query建议开启
        :param q_tf_dict: token dict
        :param q_total: query的token数量
        :return: BM25分数
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
        """ 关键词抽取
        """
        pass
