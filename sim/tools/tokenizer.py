#! -*- coding: utf-8 -*-
""" Text word segmentation tool and Tokenizer
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math

import numpy as np
import os
from collections import defaultdict
from collections import OrderedDict
from typing import Any


class Tokenizer(object):
    """ Implementation of Text-Word-Segmentation tool and Tokenizer
    """

    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                 split=" ", char_level=False, oov_token=None, document_count=0) -> None:
        """
        :param num_words: the maximum number of tokens saved, based on the freq
        :param filters: filter rules, filter all punctuation marks, tabs, newlines, etc. by default
        :param lower: whether to convert text to lowercase
        :param split: delimiter
        :param char_level: whether to use character level as token
        :param oov_token: unregistered words
        :param document_count: total text
        """

        self.word_counts = OrderedDict()  # word count in total text
        self.word_docs = defaultdict(int)  # record the number of times a token appears in the text
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = document_count  # document count
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = defaultdict(int)  # index-docs count, dict
        self.word_index = {}
        self.index_word = {}
        self.counts = list()
        self.length_average = 0.  # average document length

    def fit_on_texts(self, texts: list) -> None:
        """ Update internal vocabulary
        :param texts: text list
        :return: seq after conversion
        """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text, filters=self.filters, lower=self.lower, split=self.split)

            self.length_average += len(seq)
            text_tf_dict = dict()
            for w in seq:
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
                text_tf_dict[w] = text_tf_dict.get(w, 0) + 1

            self.counts.append(text_tf_dict)

            for w in set(seq):
                self.word_docs[w] += 1

        self.length_average /= self.document_count
        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)

        # put oov_token at the beginning of the vocabulary
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # 0 as a reserved index
        self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts: list) -> list:
        """ Convert the text into a token seq. Note that only the first num_words
            tokens will be converted, and the rest will be converted to oov_token words
        :param texts: text list
        :return: seq after conversion
        """
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts: list):
        """ Generator that converts text seq into token seq
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text, filters=self.filters, lower=self.lower, split=self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

    def sequences_to_texts(self, sequences: list) -> list:
        """ Generator that converts token seq into text seq
        :param sequences: token seq
        :return: text after conversion
        """
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences: list):
        """ Convert the token seq into a text. Note that
            only the first num_words tokens will be converted
        :param sequences: token seq
        :return: text after conversion
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = ' '.join(vect)
            yield vect

    def get_tf_idf_score(self, query: list, index: int, e: int = 0.5) -> float:
        """ Calculate the tf-idf score between the token seq and the seq of specified index
        :param query: token seq
        :param index: the index of the specified seq
        :param e: adjustable factor
        :return: tf-idf score
        """
        score = 0.
        total = sum(self.counts[index].values())
        for token in query:
            if token not in self.counts[index]:
                continue
            idf = math.log((self.document_count + e) / (self.word_docs[token] + e))
            score += (self.counts[index][token] / total) * idf

        return score

    def tf_idf_retrieval(self, query: list, top_k: int = 0, e: int = 0.5) -> list:
        """ Retrieve the top-k seq with the highest tf-idf score in the list. When top-k is 0,
            return the tf-idf scores of all seq in the text list. Sort in desc.
        :param query: token list
        :param top_k: the num of return
        :param e: adjustable factor
        :return: tf-idf scores
        """
        scores = list()
        for i in range(self.document_count):
            node = (i, self.get_tf_idf_score(query=query, index=i, e=e))
            scores.append(node)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores if top_k == 0 else scores[:top_k]

    def get_bm25_score(self, query: list, index: int, q_tf_dict: dict = None, q_total: int = 0,
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
            idf = math.log((self.document_count - self.word_docs[token] + e) / (self.word_docs[token] + e))
            tf_td = self.counts[index][token] / d_total
            sim_td = ((k1 + 1) * tf_td) / (k1 * (1 - b + b * d_total / self.length_average) + tf_td)

            sim_tq = 1.
            if if_tq:
                tf_tq = q_tf_dict[token] / q_total
                sim_tq = ((k2 + 1) * tf_tq) / (k2 + tf_tq)

            score += idf * sim_td * sim_tq

        return score

    def bm25_idf_retrieval(self, query: list, top_k: int = 0, if_tq: bool = True,
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
            node = (i, self.get_bm25_score(query=query, index=i, q_tf_dict=q_tf_dict,
                                           q_total=q_total, if_tq=if_tq, e=e, b=b, k1=k1, k2=k2))
            scores.append(node)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores if top_k == 0 else scores[:top_k]

    def get_config(self) -> dict:
        """ Get the configuration dict of the Tokenizer
        """
        json_word_counts = json.dumps(self.word_counts)
        json_word_docs = json.dumps(self.word_docs)
        json_index_docs = json.dumps(self.index_docs)
        json_word_index = json.dumps(self.word_index)
        json_index_word = json.dumps(self.index_word)
        json_counts = json.dumps(self.counts)

        return {
            'length_average': self.length_average,
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'char_level': self.char_level,
            'oov_token': self.oov_token,
            'document_count': self.document_count,
            'word_counts': json_word_counts,
            'word_docs': json_word_docs,
            'index_docs': json_index_docs,
            'index_word': json_index_word,
            'word_index': json_word_index,
            'counts': json_counts,
        }

    def to_json(self, **kwargs) -> str:
        """ Convert the Tokenizer configuration to json format and return
        """
        config = self.get_config()
        tokenizer_config = {
            'class_name': self.__class__.__name__,
            'configs': config
        }
        return json.dumps(tokenizer_config, **kwargs)


def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ") -> list:
    """ Convert text to word list
    :param text: test list
    :param filters: filter rules, filter all punctuation marks, tabs, newlines, etc. by default
    :param lower: whether to convert text to lowercase
    :param split: delimiter
    """
    if lower:
        text = text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def pad_sequences(sequences, max_len=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.) -> np.ndarray:
    """ Fill the seq, if the maximum length is not specified,
        the longest length in the seq will be used by default
    :param sequences: The seq that needs to be filled
    :param max_len: maximum length
    :param dtype:
    :param padding: filling type, pre is in front, post is in back
    :param truncating: truncating type, pre is in front, post is in back
    :param value: filling value, float or string
    :return: shape (len(sequences), max_len)
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if max_len is None:
        max_len = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, max_len) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def tokenizer_from_json(json_string) -> Tokenizer:
    """ Convert the configuration serialized json into a Tokenizer instance
    :param json_string: json string
    :return: Tokenizer
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('configs')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))
    length_average = int(config.pop('length_average'))
    counts = json.loads(config.pop('counts'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word
    tokenizer.length_average = length_average
    tokenizer.counts = counts

    return tokenizer


def load_tokenizer(dict_path: str) -> Tokenizer:
    """ Load Tokenizer through dict
    :param dict_path: dict path
    :return tokenizer: 分词器
    """
    if not os.path.exists(dict_path):
        print("dict not found, please try again")
        exit(0)

    with open(dict_path, "r", encoding="utf-8") as dict_file:
        json_string = dict_file.read().strip().strip("\n")
        tokenizer = tokenizer_from_json(json_string)

    return tokenizer


class Segment(object):
    """ Word segmentation tool
    """

    def __init__(self, model: str = "jieba"):
        """ use jieba by default
        :param model: the model fo word segmentation tool, support jieba, lac, pkuseg
        """
        self.model = model
        self.seg = None

        if model == "jieba":
            import jieba
            self.seg = jieba
        elif model == "lac":
            from LAC import LAC
            self.seg = LAC(mode="seg")
        elif model == "pkuseg":
            import pkuseg
            self.seg = pkuseg.pkuseg()

    def cut(self, sentence: str, split: str = " ") -> Any:
        """ Word segmentation of text
        :param sentence:
        :param split:
        :return:
        """
        if self.model == "jieba":
            return split.join(self.seg.cut(sentence))
        elif self.model == "lac":
            return split.join(self.seg.run(sentence))
        elif self.model == "pkuseg":
            return split.join(self.seg.cut(sentence))
