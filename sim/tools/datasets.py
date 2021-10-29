#! -*- coding: utf-8 -*-
""" Corpus Preprocess
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from sim.tools.tokenizer import pad_sequences
from sim.tools.tokenizer import Segment
from sim.tools.tokenizer import Tokenizer


def text_pair_to_token_id(file_path: str, save_path: str, split: str = "\t", seg_model: str = "jieba",
                          pad_max_len: int = None, padding: str = 'post', truncating: str = 'post',
                          value: int = 0, print_count: int = 1000, tokenizer: Tokenizer = None) -> Tokenizer:
    """ 将Text pair转换为token id
    :param file_path: 未处理的文本数据路径，文本格式: <text1><split><text2><split><label>
    :param save_path: 保存处理后的数据路径
    :param split: text pair的分隔符
    :param seg_model: 分词工具model，支付jieba, lac, pkuseg
    :param pad_max_len: padding size
    :param padding: 填充类型，pre在前，post在后
    :param truncating: 截断类型，pre在前，post在后
    :param value: 填充值类型，float或者是string
    :param print_count: 处理print_count数量数据打印日志
    :param tokenizer: 分词器
    :return: 分词器
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Raw text file not found")

    count, segment, text1s, text2s, labels = 0, None, list(), list(), list()

    if seg_model:
        segment = Segment(model=seg_model)
    with open(file_path, "r", encoding="utf-8") as raw_file, open(save_path, "a", encoding="utf-8") as save_file:
        for line in raw_file:
            line = line.strip().strip("\n")
            if line == "" or len(line.split(split)) != 3:
                continue

            pair = line.split(split)
            if seg_model:
                pair[0] = segment.cut(pair[0])
                pair[1] = segment.cut(pair[1])
            text1s.append(pair[0])
            text2s.append(pair[1])
            labels.append(pair[2])

            count += 1
            if count % print_count == 0:
                print("\r{} text-pairs processed".format(count), end="", flush=True)

        if not tokenizer:
            tokenizer = Tokenizer(oov_token="[UNK]")
            tokenizer.fit_on_texts(texts=text1s + text2s)

        text1s = tokenizer.texts_to_sequences(texts=text1s)
        text2s = tokenizer.texts_to_sequences(texts=text2s)

        if pad_max_len:
            text1s = pad_sequences(sequences=text1s, max_len=pad_max_len,
                                   padding=padding, truncating=truncating, value=value)
            text2s = pad_sequences(sequences=text2s, max_len=pad_max_len,
                                   padding=padding, truncating=truncating, value=value)

        print("\nWrite in...")
        for index, (text1, text2, label) in enumerate(zip(text1s, text2s, labels)):
            save_file.write(
                "{}{}{}{}{}\n".format(" ".join(map(str, text1)), split, " ".join(map(str, text2)), split, label))

            if index % print_count == 0:
                print("\r{} text-pairs processed".format(index), end="", flush=True)

    return tokenizer


def datasets_generator(file_path: str, batch_size: int, split: str = "\t"):
    """ Datasets generator
    :param file_path: 已分词的数据路径
    :param batch_size: batch size
    :param split: token pair的分隔符
    :return: None
    """
    with open(file_path, "r", encoding="utf-8") as file:
        datasets = file.readlines()

        np.random.shuffle(datasets)

        steps = len(datasets) // batch_size
        for i in range(steps):
            input1, input2, label = list(), list(), list()
            for sample in datasets[i:i + batch_size]:
                sample = sample.split(split)
                input1.append(list(map(int, sample[0].split(" "))))
                input2.append(list(map(int, sample[1].split(" "))))
                label.append(int(sample[2]))

            yield np.asarray(input1), np.asarray(input2), np.asarray(label), steps
