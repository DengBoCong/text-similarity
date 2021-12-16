#! -*- coding: utf-8 -*-
""" Data Format
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import sys
from typing import Any
from typing import NoReturn


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
                    line = str(line, "utf-8")
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
