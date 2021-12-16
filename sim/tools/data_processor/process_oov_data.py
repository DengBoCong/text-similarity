#! -*- coding: utf-8 -*-
""" process oov text
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
from typing import Any


def process_oov_record(record: Any,
                       normal_vocab: dict,
                       idmap: dict,
                       min_freq: int = 5,
                       min_oov_word_idx: int = 35000,
                       split: str = " ") -> tuple:
    """将相关的token转换成词频idx
    :param record: (text_a, text_b, label)
    :param normal_vocab: 词频dict
    :param idmap: token idx
    :param min_freq: 最小保留词频
    :param min_oov_word_idx: oov起始idx
    :param split: text文本分隔符
    """
    tokens_a = record[0].split(split)
    tokens_b = record[1].split(split)

    oov_word_map, cur_oov_idx = {}, min_oov_word_idx
    for tokens in [tokens_a, tokens_b]:
        for i in range(len(tokens)):
            if normal_vocab.get(tokens[i], 0) < min_freq:
                if tokens[i] not in oov_word_map:
                    oov_word_map[tokens[i]] = str(cur_oov_idx)
                    cur_oov_idx += 1
                tokens[i] = oov_word_map[tokens[i]]
            else:
                tokens[i] = idmap[tokens[i]]

    return " ".join(tokens_a), " ".join(tokens_b)


def construct_normal_vocab(train_record_list: Any,
                           output_file_path: str,
                           output_idmap_file_path: str,
                           split: str = " ") -> tuple:
    """对整个训练数据集的词频进行统计，并移除一定的低词频token，生成词典
    :param train_record_list: record list，一个record一个element
    :param output_file_path: 词频dict
    :param output_idmap_file_path: token index, 按照词频数从高到低
    :param split: text文本分隔符
    """
    word_ct = Counter()
    normal_vocab = {}
    for _, record in tqdm(train_record_list):
        word_ct.update(record[0].split(split) + record[1].split(split))
    idmap = {}
    for idx, (word, num) in enumerate(word_ct.most_common()):
        normal_vocab[word] = num
        idmap[word] = str(idx)
    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(normal_vocab, file, ensure_ascii=False, indent=4)
    with open(output_idmap_file_path, "w", encoding="utf-8") as file:
        json.dump(idmap, file, ensure_ascii=False, indent=4)

    return normal_vocab, idmap


def process_oov_records(record_list: Any,
                        normal_vocab: dict,
                        idmap: dict,
                        min_freq: int = 5,
                        min_oov_word_idx: int = 35000,
                        split: str = " ") -> list:
    """将token转化成词频token，并转化oov词
    :param record_list: record list，一个record一个element
    :param normal_vocab: 词频dict
    :param idmap: token idx
    :param min_freq: 最小保留词频
    :param min_oov_word_idx: oov起始idx
    :param split: text文本分隔符
    """
    result = []
    for idx, record in tqdm(record_list):
        text_a, text_b = process_oov_record(record=record, normal_vocab=normal_vocab, idmap=idmap,
                                            min_freq=min_freq, min_oov_word_idx=min_oov_word_idx, split=split)
        result.append((text_a, text_b, record[2]))

    return result


def process_oov_file(input_file_path: str,
                     output_file_path: str,
                     normal_vocab: dict,
                     idmap: dict,
                     min_freq: int = 5,
                     min_oov_word_idx: int = 35000,
                     header: str = None,
                     delimiter: str = "\t",
                     split: str = " ",
                     data_format: str = "normal") -> list:
    """对原始数据集的token转化成token id，并转化oov
    :param input_file_path: 数据集文件路径
    :param output_file_path: 输出处理好的数据文件路径
    :param normal_vocab: 词频dict
    :param idmap: token idx
    :param min_freq: 最小保留词频
    :param min_oov_word_idx: oov起始idx
    :param header: 文件是否有列头
    :param delimiter: 句间分隔符
    :param split: 如果数据集是一行一个sample，则split必传，作为text之间的分隔符
    :param data_format: 样本的数据排列格式
            normal: (text_a, text_b, label, ...)
    """
    records = pd.read_csv(filepath_or_buffer=input_file_path, header=header, delimiter=delimiter)
    result = process_oov_records(record_list=records.iterrows(), normal_vocab=normal_vocab, idmap=idmap,
                                 min_freq=min_freq, min_oov_word_idx=min_oov_word_idx, split=split)
    with open(output_file_path, "w", encoding="utf-8") as file:
        for record in result:
            file.write(f"{record[0]}\t{record[1]}\t{record[2]}\n")

    return result
