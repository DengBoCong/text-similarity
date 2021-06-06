#! -*- coding: utf-8 -*-
""" Coding Tools
"""

# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

import logging
import datetime


def counter(sentences):
    """ 计算分词句子列表的词频次

    :param sentences: 分词句子列表
    :return 词频次列表
    """
    word_counts = []
    for sentence in sentences:
        count = {}
        for word in sentence:
            if not count.get(word):
                count.update({word: 1})
            elif count.get(word):
                count[word] += 1
        word_counts.append(count)
    return word_counts


def load_log(filename=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")):
    """ 加载项目运行日志器

    :param filename: 日志文件，默认当前时间命名
    """
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    log_file = "{}.log"
