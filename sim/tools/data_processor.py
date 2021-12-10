#! -*- coding: utf-8 -*-
""" Corpus Preprocess
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import os
import sys
from sim.tools.settings import RNN_BASE_LOG_FILE_PATH
from sim.tools.tokenizer import pad_sequences
from sim.tools.tokenizer import Segment
from sim.tools.tokenizer import Tokenizer
from sim.tools.tools import get_logger
from typing import Any
from typing import NoReturn

logger = get_logger(name="datasets", file_path=RNN_BASE_LOG_FILE_PATH)


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

        logger.info("{} text-pairs processed".format(count))

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

        logger.info("Begin write in")
        for index, (text1, text2, label) in enumerate(zip(text1s, text2s, labels)):
            save_file.write(
                "{}{}{}{}{}\n".format(" ".join(map(str, text1)), split, " ".join(map(str, text2)), split, label))

            if index % print_count == 0:
                print("\r{} text-pairs processed".format(index), end="", flush=True)

        logger.info("Finish write in")

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


class ClassificationInputSample(object):
    """seq分类的输入样本"""

    def __init__(self, guid: Any, text_a: str, text_b: str = None, label: str = None) -> NoReturn:
        """构建ClassificationInputSample
        :param guid: 样本唯一ID
        :param text_a: 原始seq文本a
        :param text_b: 原始seq文本b，可选，针对文本对分类任务
        :param label: 样本标签，提供给train/dev，test可选
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """数据的feature集合"""

    def __init__(self, input_ids: list, input_mask: list, segment_ids: list, label_id: list) -> NoReturn:
        """构建InputFeatures
        :param input_ids:
        :param input_mask:
        :param segment_ids:
        :param label_id:
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """将数据转化为seq分类数据集的基类"""

    def get_train_samples(self, data_path: str) -> list:
        """获取训练集的ClassificationInputSample集合
        :param data_path: 训练集文件路径
        """
        raise NotImplementedError("get_train_samples must be implemented")

    def get_dev_samples(self, data_path: str) -> list:
        """获取验证集的ClassificationInputSample集合
        :param data_path: 验证集文件路径
        """
        raise NotImplementedError("get_dev_samples must be implemented")

    def get_test_samples(self, data_path: str) -> list:
        """获取验证集的ClassificationInputSample集合
        :param data_path: 测试集文件路径
        """
        raise NotImplementedError("get_test_samples must be implemented")

    def get_labels(self) -> list:
        """获取当前数据集的labels list"""
        raise NotImplementedError("get_labels must be implemented")

    @classmethod
    def _read_file(cls, input_file: str, quote_char: str = None) -> list:
        """读取以\t分隔的文件
        :param input_file:
        :param quote_char: 单字符，用于包住含有特殊字符的字段
        """
        with open(input_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t", quotechar=quote_char)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = unicode(line, "utf-8")
                lines.append(line)
            return lines


class MRPCProcessor(DataProcessor):
    """MRPC数据集（GLUE版本）"""

    def get_train_samples(self, data_path: str) -> list:
        return self._create_samples(lines=self._read_file(data_path), set_type="train")

    def get_dev_samples(self, data_path: str) -> list:
        return self._create_samples(lines=self._read_file(data_path), set_type="dev")

    def get_test_samples(self, data_path: str) -> list:
        return self._create_samples(lines=self._read_file(data_path), set_type="test")

    def get_labels(self) -> list:
        return ["0", "1"]

    def _create_samples(self, lines: list, set_type: str) -> list:
        """创建ClassificationInputSample
        :param lines: 一行一个sample
        :param set_type: 数据集类型
        :return: ClassificationInputSample list
        """
        samples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            samples.append(ClassificationInputSample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return samples


class LCQMCProcessor(DataProcessor):
    """LCQMC数据集"""

    def get_train_samples(self, data_path: str) -> list:
        return self._create_samples(lines=self._read_file(data_path), set_type="train")

    def get_dev_samples(self, data_path: str) -> list:
        return self._create_samples(lines=self._read_file(data_path), set_type="dev")

    def get_test_samples(self, data_path: str) -> list:
        return self._create_samples(lines=self._read_file(data_path), set_type="test")

    def get_labels(self) -> list:
        return ["0", "1"]

    def _create_samples(self, lines: list, set_type: str) -> list:
        """创建ClassificationInputSample
        :param lines: 一行一个sample
        :param set_type: 数据集类型
        :return: ClassificationInputSample list
        """
        samples = []
        for (i, line) in enumerate(lines):
            if i == 0 and "label" in line:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            samples.append(ClassificationInputSample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return samples
