#! -*- coding: utf-8 -*-
""" BM25及相关方法实现
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sim.base import IdfBase


class BM25(IdfBase):
    """ BM25
    """

    def __init__(self, tokens_list: list = None, file_path: str = None, file_list: list = None, split: str = None):
        """ tokens_list、file_path、file_list三者传其一，tokens_list为文本列表时，split必传
        :param tokens_list: 已经分词的文本列表或token列表
        :param file_path: 已分词文本列表文件路径，一行一个文本
        :param file_list: 已分词的文本列表文件路径列表，一行一个文本
        :param split: 文本分隔符，list模式不传则每个element视为list，file模式必传
        :return: None
        """
        IdfBase.__init__(self, tokens_list, file_path, file_list, split)
