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
import unicodedata
import numpy as np
import os
import re
import sentencepiece as spm
from collections import defaultdict
from collections import OrderedDict
from typing import Any
from typing import NoReturn


class Tokenizer(object):
    """ 文本分词工具及Tokenizer
    """

    def __init__(self,
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=" ",
                 char_level=False,
                 oov_token=None,
                 document_count=0) -> None:
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

    def get_bm25_score(self,
                       query: list,
                       index: int,
                       q_tf_dict: dict = None,
                       q_total: int = 0,
                       if_tq: bool = True,
                       e: int = 0.5,
                       b=0.75,
                       k1=2,
                       k2=1.2) -> float:
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

    def bm25_idf_retrieval(self,
                           query: list,
                           top_k: int = 0,
                           if_tq: bool = True,
                           e: int = 0.5,
                           b=0.75,
                           k1=2,
                           k2=1.2) -> list:
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


def pad_sequences(sequences,
                  max_len=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.) -> np.ndarray:
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
        raise FileNotFoundError("dict not found, please try again")

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


def truncate_sequences(max_len: int, indices: Any, *sequences) -> list:
    """截断总长度至不超过max_len
    :param max_len: 最大长度
    :param indices: int/list
    :param sequences: 序列list
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > max_len:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


class BertTokenizerBase(object):
    """Bert分词器基类
    """

    def __init__(self,
                 token_start: Any = "[CLS]",
                 token_end: Any = "[SEP]",
                 pre_tokenize: Any = None,
                 token_translate: dict = None):
        """
        :param token_start: 起始token
        :param token_end: 结束token
        :param pre_tokenize: 外部传入的分词函数，用作对文本进行预分词。如果传入
                            pre_tokenize，则先执行pre_tokenize(text)，然后在它
                            的基础上执行原本的tokenize函数
        :param token_translate: 映射字典，主要用在tokenize之后，将某些特殊的token替换为对应的token
        """
        self._token_pad = "[PAD]"
        self._token_unk = "[UNK]"
        self._token_mask = "[MASK]"
        self._token_start = token_start
        self._token_end = token_end
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {v: k for k, v in self._token_translate.items()}

    def tokenize(self, text: str, max_len: int = None) -> list:
        """分词
        :param text: 切词文本
        :param max_len: 填充长度
        """
        tokens = [self._token_translate.get(token) or token for token in self._tokenize(text)]

        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if max_len is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences(max_len, -index, tokens)

        return tokens

    def token_to_id(self, token: str) -> Any:
        """token转换为对应的id
        :param token: token
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens: list) -> list:
        """token序列转换为对应的id序列
        :param tokens: token list
        """
        return [self.token_to_id(token) for token in tokens]

    def encode(self,
               first_text: Any,
               second_text: Any = None,
               max_len: int = None,
               pattern: str = "S*E*E",
               truncate_from: Any = "post") -> tuple:
        """输出文本对应token id和segment id
        :param first_text: str/list
        :param second_text: str/list
        :param max_len: 最大长度
        :param pattern: pattern
        :param truncate_from: 填充位置，str/int
        """
        if isinstance(first_text, str):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif isinstance(second_text, str):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if max_len is not None:
            if truncate_from == "post":
                index = -int(self._token_end is not None) - 1
            elif truncate_from == "pre":
                index = int(self._token_start is not None)
            else:
                index = truncate_from

            if second_text is not None and pattern == "S*E*E":
                max_len += 1
            truncate_sequences(max_len, index, first_tokens, second_tokens)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            if pattern == "S*E*E":
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def id_to_token(self, i: int) -> Any:
        """id转为对应个token
        :param i: id
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids: list) -> list:
        """id序列转为对应的token序列
        :param ids: id list
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids: list) -> Any:
        """转为可读文本
        :param ids: id list
        """
        raise NotImplementedError

    def _tokenize(self, text: str) -> list:
        """
        :param text: 切词文本
        """
        raise NotImplementedError


class BertTokenizer(BertTokenizerBase):
    """Bert原生分词器
    """

    def __init__(self,
                 token_dict: Any,
                 do_lower_case: bool = False,
                 word_max_len: int = 200,
                 **kwargs):
        """
        :param token_dict: 映射字典或其文件路径
        :param do_lower_case: 小写化
        :param word_max_len: 最大长度
        """
        super(BertTokenizer, self).__init__(**kwargs)
        if isinstance(token_dict, str):
            token_dict = self.load_vocab(token_dict)

        self._do_lower_case = do_lower_case
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._vocab_size = len(token_dict)
        self._word_max_len = word_max_len

        for token in ["pad", "unk", "mask", "start", "end"]:
            try:
                _token_id = token_dict[getattr(self, f"_token_{token}")]
                setattr(self, f"_token_{token}_id", _token_id)
            except Exception as e:
                print(str(e))

    def token_to_id(self, token: str) -> Any:
        """token转换为对应的id
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i: int) -> Any:
        """id转换为对应的token
        """
        return self._token_dict_inv[i]

    def decode(self, ids: list, tokens: list = None) -> Any:
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self.is_special(token)]

        text, flag = "", False
        for i, token in enumerate(tokens):
            if token[:2] == "##":
                text += token[2:]
            elif len(token) == 1 and self.is_cjk_character(token):
                text += token
            elif len(token) == 1 and self.is_punctuation(token):
                text += token
                text += " "
            elif i > 0 and self.is_cjk_character(text[-1]):
                text += token
            else:
                text += " "
                text += token

        text = re.sub(" +", " ", text)
        text = re.sub("' (re|m|s|t|ve|d|ll) ", "'\\1 ", text)
        punctuation = self.cjk_punctuation() + "+-/={(<["
        punctuation_regex = "|".join([re.escape(p) for p in punctuation])
        punctuation_regex = "(%s) " % punctuation_regex
        text = re.sub(punctuation_regex, "\\1", text)
        text = re.sub("(\d\.) (\d)", "\\1\\2", text)

        return text.strip()

    def _tokenize(self, text: str, pre_tokenize: bool = True) -> list:
        """基本分词函数
        """
        if self._do_lower_case:
            text = BertTokenizer.lowercase_and_normalize(text)

        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))

            return tokens

        spaced = ""
        for ch in text:
            if self.is_punctuation(ch) or self.is_cjk_character(ch):
                spaced += " " + ch + " "
            elif self.is_space(ch):
                spaced += " "
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self.is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self.word_piece_tokenize(word))

        return tokens

    def word_piece_tokenize(self, word: str):
        """word内分成subword
        """
        if len(word) > self._word_max_len:
            return [word]

        tokens, start, end = [], 0, 0
        while start < len(word):
            end = len(word)
            while end > start:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self._token_dict:
                    break
                end -= 1

            if start == end:
                return [word]
            else:
                tokens.append(sub)
                start = end

        return tokens

    @staticmethod
    def lowercase_and_normalize(text: str):
        """转小写，并进行简单的标准化
        """
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join([ch for ch in text if unicodedata.category(ch) != "Mn"])
        return text

    @staticmethod
    def stem(token: str) -> str:
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == "##":
            return token[2:]
        else:
            return token

    @staticmethod
    def is_space(ch) -> bool:
        """空格类字符判断
        """
        return ch == " " or ch == "\n" or ch == "\r" or ch == "\t" or unicodedata.category(ch) == "Zs"

    @staticmethod
    def is_punctuation(ch) -> bool:
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith("P")

    @staticmethod
    def cjk_punctuation() -> str:
        return u"\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002"

    @staticmethod
    def is_cjk_character(ch) -> bool:
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def is_control(ch: Any) -> bool:
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def is_special(ch: Any) -> bool:
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def is_redundant(token: str) -> bool:
        """判断该token是否冗余（默认情况下不可能分出来）
        """
        if len(token) > 1:
            for ch in BertTokenizer.stem(token):
                if BertTokenizer.is_cjk_character(ch) or BertTokenizer.is_punctuation(ch):
                    return True

    def rematch(self, text: str, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

    @staticmethod
    def load_vocab(dict_path: str,
                   encoding: str = "utf-8",
                   simplified: bool = False,
                   startswith: list = None) -> Any:
        """从bert的词典文件中读取词典
        :param dict_path: 字典文件路径
        :param encoding: 编码格式
        :param simplified: 是否过滤冗余部分token
        :param startswith: 附加在起始的list
        """
        token_dict = {}
        with open(dict_path, "r", encoding=encoding) as reader:
            for line in reader:
                token = line.split()
                token = token[0] if token else line.strip()
                token_dict[token] = len(token_dict)

        # 是否过滤冗余部分token
        if simplified:
            new_token_dict, keep_tokens = {}, []
            startswith = startswith or []
            for t in startswith:
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])

            for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
                if t not in new_token_dict and not BertTokenizer.is_redundant(t):
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

            return new_token_dict, keep_tokens
        else:
            return token_dict

    @staticmethod
    def save_vocab(dict_path: str, token_dict: dict, encoding: str = "utf-8") -> NoReturn:
        """将词典（比如精简过的）保存为文件
        """
        with open(dict_path, "w+", encoding=encoding) as writer:
            for k, v in sorted(token_dict.items(), key=lambda s: s[1]):
                writer.write(k + "\n")


class SpTokenizer(BertTokenizerBase):
    """基于SentencePiece模型的封装，使用上跟BERTTokenizer基本一致
    """

    def __init__(self, sp_model_path: str, **kwargs):
        """
        :param sp_model_path: SentencePiece模型模型路径
        """
        super(SpTokenizer, self).__init__(**kwargs)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)
        self._token_pad = self.sp_model.IdToPiece(self.sp_model.pad_id())
        self._token_unk = self.sp_model.IdToPiece(self.sp_model.unk_id())
        self._vocab_size = self.sp_model.GetPieceSize()

        for token in ["pad", "unk", "mask", "start", "end"]:
            try:
                _token = getattr(self, f"_token_{token}")
                _token_id = self.sp_model.PieceToId(_token)
                setattr(self, f"_token_{token}_id", _token_id)
            except Exception as e:
                print(str(e))

    def token_to_id(self, token: str) -> Any:
        """token转换为对应的id
        """
        return self.sp_model.PieceToId(token)

    def id_to_token(self, i: int) -> Any:
        """id转换为对应的token
        """
        if i < self._vocab_size:
            return self.sp_model.IdToPiece(i)
        else:
            return ""

    def decode(self, ids: list) -> Any:
        """转为可读文本
        """
        tokens = [self._token_translate_inv.get(token) or token for token in self.ids_to_tokens(ids)]
        text = self.sp_model.DecodePieces(tokens)

        if isinstance(text, bytes):
            return text.decode(encoding="utf-8", errors="ignore")

        return text

    def _tokenize(self, text: str) -> list:
        """基本分词函数
        """
        if self._pre_tokenize is not None:
            text = " ".join(self._pre_tokenize(text))

        tokens = self.sp_model.EncodeAsPieces(text)
        return tokens

    def is_special(self, ch: Any):
        """判断是不是有特殊含义的符号
        """
        return self.sp_model.IsControl(ch) or self.sp_model.IsUnknown(ch) or self.sp_model.IsUnused(ch)

    def is_decodable(self, i: Any):
        """判断是否应该被解码输出
        """
        return (i < self._vocab_size) and not self.is_special(i)
