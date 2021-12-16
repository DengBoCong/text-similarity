#! -*- coding: utf-8 -*-
""" process cipher text
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pandas as pd
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from typing import NoReturn

logger = get_logger(name="processor", file_path=RUNTIME_LOG_FILE_PATH)


def construct_vocab(tokens: list, vocab_file_path: str):
    """构建Unicode中字符-索引映射表
    :param tokens: unicode字符列表
    :param vocab_file_path: 映射表文件路径
    """
    vocab = {"num2token": {}, "token2num": {}}
    num_id = 0
    for token in tokens:
        if chr(token) not in vocab["token2num"]:
            vocab["token2num"][num_id] = chr(token)
            vocab["num2token"][chr(token)] = num_id
            num_id += 1
    with open(vocab_file_path, "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)
    return vocab


def convert_record_style(input_file_path: str,
                         vocab: dict,
                         output_file_path: str,
                         header: str = None,
                         delimiter: str = "\t",
                         split: str = " ") -> NoReturn:
    """这里将密文转换成Unicode中存在的字符（汉字或英文字符）
    :param input_file_path: 密文文件路径，密文文件必须一行一个sample，且(text_a, text_b, label)
    :param vocab: 字符-索引映射表
    :param output_file_path: 转化明文文件路径
    :param header: 文件是否有列头
    :param delimiter: 句间分隔符
    :param split: 如果数据集是一行一个sample，则split必传，作为text之间的分隔符
    """
    records = pd.read_csv(filepath_or_buffer=input_file_path, header=header, delimiter=delimiter)

    def convert_str_style(input_str):
        token_ids = [int(item) for item in input_str.split(split) if item]
        tokens = [vocab["num2token"][idx] for idx in token_ids]
        return "".join(tokens)

    records[0] = records[0].apply(convert_str_style)
    records[1] = records[1].apply(convert_str_style)
    records.to_csv(path_or_buf=output_file_path, header=header, sep=delimiter, index=False)
