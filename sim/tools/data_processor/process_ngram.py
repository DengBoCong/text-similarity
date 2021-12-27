#! -*- coding: utf-8 -*-
""" process ngram
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl
from collections import Counter
from gensim.models import KeyedVectors
from nltk import ngrams
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from tqdm import tqdm
from transformers import BertTokenizer
from typing import NoReturn

logger = get_logger(name="processor", file_path=RUNTIME_LOG_FILE_PATH)


def construct_ngram_dict(file_path: str,
                         save_path: str,
                         min_freq: int = 10,
                         max_ngram: int = 3) -> NoReturn:
    """构造ngram的映射表
    :param file_path: 数据集文件路径，一行一个text
    :param save_path: 映射表保存路径
    :param min_freq: 最小保留词频
    :param max_ngram:
    """
    token_counter = Counter()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            line = line.strip().strip("\n")
            if line == "":
                continue
            for ngram_num in range(2, max_ngram + 1):
                token_counter.update(["".join(item) for item in ngrams(line, ngram_num)])
        ngram_dict = {}
        for k, v in tqdm(token_counter.most_common()):
            if v >= min_freq:
                ngram_dict[k] = v
            else:
                break

    logger.info(f'ngram_words: {len(ngram_dict)}')
    output_ngram_words = {}
    for k, v in sorted([(k, v) for k, v in ngram_dict.items()], key=lambda x: len(x[0])):
        seq_len = len(k)
        for start_pos in range(0, seq_len):
            for end_pos in range(start_pos + 1, seq_len):
                if k[start_pos:end_pos] in output_ngram_words and \
                        output_ngram_words[k[start_pos:end_pos]] == v and end_pos - start_pos > 1:
                    output_ngram_words.pop(k[start_pos:end_pos])
        output_ngram_words[k] = v

    logger.info(f'ngram_words: {len(output_ngram_words)}')

    with open(save_path, "w", encoding="utf-8") as file:
        pkl.dump(output_ngram_words, file)


def convert_to_ids(tokens: str, tokenizer: BertTokenizer) -> tuple:
    """tokenizer转化tokens
    :param tokens: 待转的tokens
    :param tokenizer: tokenizer
    """
    return tuple(tokenizer.convert_tokens_to_ids(item) for item in tokens)


def get_similar_words(token: str, word2vec: KeyedVectors) -> list:
    """获取word2vec中最相似的前15个token
    :param token: token
    :param word2vec: word2vec model
    """
    results = word2vec.similar_by_word(token, topn=100)
    sim_words = []
    for idx, item in enumerate(results):
        if len(sim_words) >= 15:
            break
        if len(item[0]) == len(token) and item[1] > 0.68:
            sim_words.append(item[0])

    return sim_words


def construct_ngram_meta_info(w2v_model_path: str,
                              ngram_file: str,
                              tokenizer_path: str,
                              output_meta_path: str) -> NoReturn:
    """构建ngram的元信息
    :param w2v_model_path: word2vec的模型路径
    :param ngram_file: ngram文件路径
    :param tokenizer_path: tokenizer路径
    :param output_meta_path: 保存路径
    """
    ngram_dict = {
        "ngrams": {i: [] for i in range(1, 5)},
        "ngrams_set": {},
        "sim_ngrams": {}
    }

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    word2vec = KeyedVectors.load(w2v_model_path)
    ngram_words = pkl.load(open(ngram_file, "rb"))
    all_words = list(set(list(tokenizer.get_vocab().keys()) + list(ngram_words.keys())))
    ngram_dict["ngrams_set"] = set()
    for word in tqdm(all_words):
        ngram_dict["ngrams_set"].add(convert_to_ids(word, tokenizer))
        if word not in word2vec:
            continue
        ngram_dict["ngrams"][len(word)].append(convert_to_ids(word, tokenizer))
        sim_words = get_similar_words(word, word2vec)
        if sim_words:
            ngram_dict["sim_ngrams"][convert_to_ids(word, tokenizer)] = []
            for sim_word in sim_words:
                ngram_dict["sim_ngrams"][convert_to_ids(word, tokenizer)].append(convert_to_ids(sim_word, tokenizer))

    with open(output_meta_path, "wb") as file:
        pkl.dump(ngram_dict, file)
