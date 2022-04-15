#! -*- coding: utf-8 -*-
""" Coding Tools
"""

# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

import json
import re
import os
import sys
import logging
import numpy as np
from datetime import datetime
from functools import lru_cache
from logging import Logger
from typing import Any

# 设定logging基础配置
LOGGING_FORMATTER = "%(asctime)s %(module)s [line:%(lineno)d] %(levelname)s: %(message)s"
logging.basicConfig(format=LOGGING_FORMATTER, datefmt="%Y-%m-%d %H:%M:%S")


class ProgressBar(object):
    """ 进度条工具 """

    EXECUTE = "%(current)d/%(total)d %(bar)s (%(percent)3d%%) %(metrics)s"
    DONE = "%(current)d/%(total)d %(bar)s - %(time).4fs/step %(metrics)s"

    def __init__(self,
                 total: int = 100,
                 num: int = 1,
                 width: int = 30,
                 fmt: str = EXECUTE,
                 symbol: str = "=",
                 remain: str = ".",
                 output=sys.stderr):
        """
        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        assert len(symbol) == 1
        self.args = {}
        self.metrics = ""
        self.total = total
        self.num = num
        self.width = width
        self.symbol = symbol
        self.remain = remain
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d", r"\g<name>%dd" % len(str(total)), fmt)

    def __call__(self, current: int, metrics: str):
        """
        :param current: 已执行次数
        :param metrics: 附加在进度条后的指标字符串
        """
        self.metrics = metrics
        percent = current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + ">" + self.remain * (self.width - size - 1) + "]"

        self.args = {
            "total": self.total * self.num,
            "bar": bar,
            "current": current * self.num,
            "percent": percent * 100,
            "metrics": metrics
        }
        print("\r" + self.fmt % self.args, file=self.output, end="")

    def reset(self,
              total: int,
              num: int,
              width: int = 30,
              fmt: str = EXECUTE,
              symbol: str = "=",
              remain: str = ".",
              output=sys.stderr):
        """重置内部属性
        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        self.__init__(total=total, num=num, width=width, fmt=fmt,
                      symbol=symbol, remain=remain, output=output)

    def done(self, step_time: float, fmt=DONE):
        """
        :param step_time: 该时间步执行完所用时间
        :param fmt: 执行完成之后进度条格式
        """
        self.args["bar"] = "[" + self.symbol * self.width + "]"
        self.args["time"] = step_time
        # print("\r" + fmt % self.args + "\n", file=self.output, end="")
        return fmt % self.args


def get_dict_string(data: dict, prefix: str = "- ", precision: str = ": {:.4f} "):
    """将字典数据转换成key——value字符串
    :param data: 字典数据
    :param prefix: 组合前缀
    :param precision: key——value打印精度
    :return: 字符串
    """
    result = ""
    for key, value in data.items():
        result += (prefix + key + precision).format(value)

    return result


def get_logger(name: str,
               file_path: str,
               level: int = logging.INFO,
               mode: str = "a+",
               encoding: str = "utf-8",
               formatter: str = LOGGING_FORMATTER) -> Logger:
    """ 获取日志器
    :param name: 日志命名
    :param file_path: 日志文件存放路径
    :param level: 最低的日志级别
    :param mode: 读写日志文件模式
    :param encoding: 日志文件编码
    :param formatter: 日志格式
    :return: 日志器
    """
    if file_path and not file_path.endswith(".log"):
        raise ValueError("{} not a valid file path".format(file_path))

    if not os.path.exists(os.path.dirname(file_path)):
        os.mkdir(os.path.dirname(file_path))

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    if not logger.handlers:
        file_logger = logging.FileHandler(filename=file_path, mode=mode, encoding=encoding)
        file_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(formatter)
        file_logger.setFormatter(formatter)
        logger.addHandler(file_logger)

    return logger


def save_model_config(key: str, model_desc: str, model_config: dict, config_path: str) -> bool:
    """ 保存单次训练执行时，模型的对应配置
    :param key: 配置key
    :param model_desc: 模型说明
    :param model_config: 训练配置
    :param config_path: 配置文件保存路径
    :return: 执行成功与否
    """
    try:
        config_json = {}
        if os.path.exists(config_path) and os.path.getsize(config_path) != 0:
            with open(config_path, "r", encoding="utf-8") as file:
                config_json = json.load(file)

        with open(config_path, "w+", encoding="utf-8") as config_file:
            model_config["execute_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_config["model_desc"] = model_desc
            config_json[key] = model_config
            json.dump(config_json, config_file, ensure_ascii=False, indent=4)

            return True
    except Exception:
        return False


def get_model_config(key: str, config_path: str) -> dict:
    """ 保存单次训练执行时，模型的对应配置
    :param key: 配置key
    :param config_path: 配置文件路径
    :return: 模型配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError("get_model_config: Not such file {}".format(config_path))

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file).get(key, {})
    except Exception:
        return {}


def orthogonally_resize(a: np.ndarray, new_shape: Any, window: int = 2) -> Any:
    """简单的正交化缩放矩阵
    :param a: 缩放的矩阵
    :param new_shape: new shape
    :param window: 缩放比例
    """
    assert a.ndim == len(new_shape)
    slices, a_norm, w = [], np.linalg.norm(a), window
    for i, (d1, d2) in enumerate(zip(a.shape, new_shape)):
        if d1 != d2:
            k = d2 // d1 + int(d2 % d1 != 0)
            if k > 1:
                assert d1 % w == 0
                a = a.reshape(a.shape[:i] + (d1 // w, w) + a.shape[i + 1:])
                a = np.repeat(a, k, axis=i)
                a = a.reshape(a.shape[:i] + (d1 * k,) + a.shape[i + 2:])

        slices.append(np.s_[:d2])
    a = a[tuple(slices)]
    return a / np.linalg.norm(a) * a_norm


def make_log_bucket_position(relative_pos: Any, bucket_size: int, max_position: int) -> Any:
    """混入log bucket位置编码
    :param relative_pos:
    :param bucket_size: bucket size
    :param max_position: 最大位置
    """
    sign = np.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = np.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, np.abs(relative_pos))
    log_pos = np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int)
    return bucket_pos


@lru_cache(maxsize=128)
def build_relative_position_deberta(query_size, key_size, bucket_size: int = -1, max_position: int = -1):
    """DeBERTa的相对位置编码"""
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0], 1))
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.astype(np.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids[None, :]
    return rel_pos_ids


def clean_str(string):
    """
    For English Corpus
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_stopwords(stop_words_file_path_list: list) -> set:
    """加载stop words
    :param stop_words_file_path_list: 停用词文件列表
    """
    stop_words_set = set()
    for stop_words_file_path in stop_words_file_path_list:
        if not os.path.exists(stop_words_file_path):
            raise FileNotFoundError(f"`{stop_words_file_path}` not found")
        with open(stop_words_file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().strip("\n")
                if line:
                    stop_words_set.add(line)

    return stop_words_set
