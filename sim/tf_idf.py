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
        """ tokens_list、file_path and file_list must pass one, when tokens_list is a text list, split must pass
        :param tokens_list: text list or token list
        :param file_path: the path of the word segmented text file, one text per line
        :param file_list: file list, one text per line
        :param split: if the list mode is not passed, element is regarded as a list, and the file mode must be passed
        """
        super(TFIdf, self).__init__(tokens_list, file_path, file_list, split)

    def get_score(self, query: list, index: int, e: int = 0.5) -> float:
        """ Calculate the tf-idf score between the token seq and the seq of specified index
        :param query: token seq
        :param index: the index of the specified seq
        :param e: adjustable factor
        :return: tf-idf score
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
        """ Retrieve the top-k seq with the highest tf-idf score in the list. When top-k is 0,
            return the tf-idf scores of all seq in the text list. Sort in desc.
        :param query: token list
        :param top_k: the num of return
        :param e: adjustable factor
        :return: tf-idf scores, element shape: (index, score)
        """
        scores = list()
        for i in range(self.document_count):
            node = (i, self.get_score(query=query, index=i, e=e))
            scores.append(node)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores if top_k == 0 else scores[:top_k]

    def weight(self, e: int = 0.5, pad_size: int = None, padding: str = "pre",
               truncating: str = "pre", value: float = 0., d_type: str = "float32") -> Any:
        """ Perform tf-idf weighted conversion on the sentence vector in tokens_list. When pad_size is not
            passed, the corresponding weight list is returned. When pad_size is passed in (valid for other
            parameters), the complemented numpy tensor is returned.
        :param e: adjustable factor
        :param pad_size: padding size
        :param padding: filling type, pre is in front, post is in back
        :param truncating: truncating type, pre is in front, post is in back
        :param value: filling value
        :param d_type:
        :return: TF-IDF weight
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
        """ Extract keywords
        """
        pass
