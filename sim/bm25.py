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
        """ tokens_list、file_path and file_list must pass one, when tokens_list is a text list, split must pass
        :param tokens_list: text list or token list
        :param file_path: the path of the word segmented text file, one text per line
        :param file_list: file list, one text per line
        :param split: if the list mode is not passed, element is regarded as a list, and the file mode must be passed
        """
        super(BM25, self).__init__(tokens_list, file_path, file_list, split)

    def get_score(self, query: list, index: int, q_tf_dict: dict = None, q_total: int = 0,
                  if_tq: bool = True, e: int = 0.5, b=0.75, k1=2, k2=1.2) -> float:
        """ Calculate the bm25 score between the token seq and the seq of specified index
        :param query: token list
        :param index: the index of the specified seq
        :param q_tf_dict: token dict
        :param q_total: the num of tokens in the query
        :param if_tq: whether to add the relevance between the word and the query, the long query is enabled by default
        :param e: adjustable factor
        :param b: adjustable factor, (0,1)
        :param k1: adjustable factor, [1.2, 2.0]
        :param k2: adjustable factor, [1.2, 2.0]
        :return: BM25 score
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
        """ Retrieve the top-k seq with the highest bm25 score in the list. When top-k
            is 0, return the bm25 scores of all seq in the text list. Sort in desc.
        :param query: token list
        :param top_k: the num of return
        :param if_tq: whether to add the relevance between the word and the query, the long query is enabled by default
        :param e: adjustable factor
        :param b: adjustable factor, (0,1)
        :param k1: adjustable factor, [1.2, 2.0]
        :param k2: adjustable factor, [1.2, 2.0]
        :return: BM25 scores
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
        """ Perform BM25 weighted conversion on the sentence vector in tokens_list. When pad_size is not
            passed, the corresponding weight list is returned. When pad_size is passed in (valid for other
            parameters), the complemented numpy tensor is returned.
        :param e: adjustable factor
        :param b: adjustable factor, (0,1)
        :param k1: adjustable factor, [1.2, 2.0]
        :param k2: adjustable factor, [1.2, 2.0]
        :param pad_size: padding size
        :param padding: filling type, pre is in front, post is in back
        :param if_tq: whether to add the relevance between the word and the query, the long query is enabled by default
        :param truncating: truncating type, pre is in front, post is in back
        :param value: filling value
        :param d_type:
        :return: BM25 weight
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
        """ Calculate bm25 value
        :param token: The currently calculated token
        :param count: freq dict
        :param total: total num of tokens
        :param e: adjustable factor
        :param b: adjustable factor, (0,1)
        :param k1: adjustable factor, [1.2, 2.0]
        :param k2: adjustable factor, [1.2, 2.0]
        :param if_tq: whether to add the relevance between the word and the query, the long query is enabled by default
        :param q_tf_dict: token dict
        :param q_total: the num of tokens in the query
        :return: BM25 score
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
        """ Extract keywords
        """
        pass
