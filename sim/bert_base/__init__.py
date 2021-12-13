#! -*- coding: utf-8 -*-
""" Bert Base Entrance
"""
# Author: DengBoCong <bocongdeng@gmail.com>
# Reconstruct it according to https://github.com/google-research/bert
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
from argparse import ArgumentParser
from importlib import import_module
from typing import NoReturn


class BertConfig(object):
    """BertModel的配置"""

    def __init__(self,
                 vocab_size: int = 30522,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_prob_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 max_relative_position: int = 64,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 pad_token_id: int = 0,
                 use_relative_position: bool = False,
                 segment_type: str = "absolute",
                 use_mean_pooling: bool = False) -> NoReturn:
        """构建BertConfig
        :param vocab_size: 词表大小
        :param hidden_size: encoder和pool维度大小
        :param num_hidden_layers: encoder的层数
        :param num_attention_heads: encoder中的attention层的注意力头数量
        :param intermediate_size: 前馈神经网络层维度大小
        :param hidden_act: encoder和pool中的非线性激活函数
        :param hidden_dropout_prob: embedding、encoder和pool层中的全连接层dropout
        :param attention_prob_dropout_prob: attention的dropout
        :param max_position_embeddings: 绝对位置编码最大位置数
        :param max_relative_position: 相对位置编码最大位置数
        :param type_vocab_size: token_type_ids的词典大小
        :param initializer_range: truncated_normal_initializer初始化方法的stdev
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param pad_token_id: 用于padding的token id
        :param use_relative_position: 是否使用相对位置编码
        :param segment_type: 相对位置还是绝对位置
        :param use_mean_pooling: 是否增加pool输出层
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_prob_dropout_prob = attention_prob_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position = max_relative_position
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.use_relative_position = use_relative_position
        self.segment_type = segment_type
        self.use_mean_pooling = use_mean_pooling

    @classmethod
    def from_dict(cls, json_obj) -> BertConfig:
        """从字典对象中构建BertConfig
        :param json_obj: 字典对象
        :return: BertConfig
        """
        bert_config = BertConfig(vocab_size=0)
        for (key, value) in json_obj.items():
            bert_config.__dict__[key] = value

        return bert_config

    @classmethod
    def from_json_file(cls, json_file_path: str) -> BertConfig:
        """从json文件中构建BertConfig
        :param json_file_path: JSON文件路径
        :return: BertConfig
        """
        with open(json_file_path, "r", encoding="utf-8") as reader:
            return cls.from_dict(json_obj=json.load(reader))

    def to_dict(self):
        """将实例序列化为字典"""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        """将实例序列化为json字符串"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def actuator() -> NoReturn:
    pass
