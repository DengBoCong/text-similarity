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
    """ 文本分词工具及Tokenizer
    """

    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                 split=" ", char_level=False, oov_token=None, document_count=0) -> None:
        """
        :param num_words: 保存的最大token数，基于出现频率
        :param filters: 过滤规则, 默认过滤所有标点符号、制表符、换行符等
        :param lower: 是否将文本转换为小写
        :param split: 分隔符
        :param char_level: 是否以字符级作为token
        :param oov_token: 未登录词
        :param document_count: 文本总数
        """

        self.word_counts = OrderedDict()  # 总文本中词计数
        self.word_docs = defaultdict(int)  # 某个token在文本中出现的次数
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = document_count  # 文本计数
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = defaultdict(int)  # 索引-出现文本计数 词典
        self.word_index = {}
        self.index_word = {}
        self.counts = list()
        self.length_average = 0.  # 文档平均长度

    def fit_on_texts(self, texts: list) -> None:
        """ 更新内部词汇表
        :param texts: 文本列表
        :return: 转换后的seq
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

        # 将未登录词放在词汇表开头
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # 索引0作为保留索引
        self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts: list) -> list:
        """ 将文本序列转化为token序列，注意了，只有前
            num_words个token才会被转换，其余转换为oov_token词
        :param texts: 文本列表
        :return: 转换后的seq
        """
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts: list):
        """ 将文本序列转化为token序列的生成器
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
        """ 将token序列转化为文本序列的生成器
        :param sequences: token序列
        :return: 转换后的文本序列
        """
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences: list):
        """ 将token序列转化为文本序列，注意了，只有前
            num_words个token才会被转换，其余转换为token词
        :param sequences: token序列
        :return: 转换后的文本序列
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
        """ 计算文本序列与文本列表指定的文本序列的tf-idf相似度分数
        :param query: 文本序列
        :param index: 指定文本列表中的文本序列索引
        :param e: 调教系数
        :return: tf-idf分数
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
        """ 检索文本列表中tf-idf分数最高的前top-k个文本序列，当
            top-k为0时，返回文本列表中所有文本序列与指定文本序列的td-idf分数
        :param query: 文本序列
        :param top_k: 返回的数量
        :param e: 调教系数
        :return: tf-idf分数列表
        """
        scores = list()
        for i in range(self.document_count):
            node = (i, self.get_tf_idf_score(query=query, index=i, e=e))
            scores.append(node)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores if top_k == 0 else scores[:top_k]

    def get_bm25_score(self, query: list, index: int, q_tf_dict: dict = None, q_total: int = 0,
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
        """ 检索文本列表中BM25分数最高的前top-k个文本序列，当
            top-k为0时，返回文本列表中所有文本序列与指定文本序列的BM25分数
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
        """ 获取分词器的配置字典
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
        """ 将分词器相关数据转化为json格式返回
        """
        config = self.get_config()
        tokenizer_config = {
            'class_name': self.__class__.__name__,
            'configs': config
        }
        return json.dumps(tokenizer_config, **kwargs)


def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ") -> list:
    """ 讲文本转换成token序列
    :param text: 文本列表
    :param filters: 过滤规则, 默认过滤所有标点符号、制表符、换行符等
    :param lower: 是否将文本转换为小写
    :param split: 分隔符
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
    """ 填充序列，如果未指定最大长度，则默认使用序列中最长长度

    :param sequences: 需要填充的序列
    :param max_len: 最大长度
    :param dtype: 输出类型
    :param padding: 填充类型，pre在前，post在后
    :param truncating: 截断类型，pre在前，post在后
    :param value: 填充值类型，float或者是string
    :return: 形状为(len(sequences), max_len)的numpy数组
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
    """ 将Tokenizer序列化的json转化为Tokenizer实例
    :param json_string: json字符串
    :return: 分词器
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
    """ 通过字典加载tokenizer
    :param dict_path: 字典路径
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
    """ 分词工具
    """

    def __init__(self, model: str = "jieba"):
        """ 需要初始化一个分词工具的base，默认使用结巴分词
        :param model: 分词工具model，支付jieba, lac, pkuseg
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
        """ 对文本进行分词
        :param sentence: 分词文本
        :param split: 分隔符，不传则返回token列表
        :return: 分词后的token列表或文本
        """
        if self.model == "jieba":
            return split.join(self.seg.cut(sentence))
        elif self.model == "lac":
            return split.join(self.seg.run(sentence))
        elif self.model == "pkuseg":
            return split.join(self.seg.cut(sentence))
