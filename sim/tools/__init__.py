#! -*- coding: utf-8 -*-
""" Some Common Components
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
from transformers.configuration_utils import PretrainedConfig
from typing import Any
from typing import NoReturn


class BertConfig(object):
    """BertModel的配置"""

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_hidden_layers: int,
                 intermediate_size: int,
                 hidden_act: Any,
                 embedding_size: int = None,
                 attention_head_size: int = None,
                 attention_key_size: int = None,
                 max_position_embeddings: int = None,
                 max_position: int = None,
                 layer_norm_eps: float = 1e-7,
                 type_vocab_size: int = None,
                 hidden_dropout_prob: float = None,
                 attention_probs_dropout_prob: float = None,
                 shared_segment_embeddings: bool = False,
                 hierarchical_position: Any = False,
                 initializer_range: float = None,
                 use_relative_position: bool = False,
                 max_relative_positions: int = None,
                 **kwargs):
        """构建BertConfig
        :param vocab_size: 词表大小
        :param hidden_size: 隐藏层大小
        :param num_attention_heads: encoder中的attention层的注意力头数量
        :param num_hidden_layers: encoder的层数
        :param intermediate_size: 前馈神经网络层维度
        :param hidden_act: encoder和pool中的非线性激活函数
        :param embedding_size: 词嵌入大小
        :param attention_head_size: Attention中V的head_size
        :param attention_key_size: Attention中Q,K的head_size
        :param max_position_embeddings: 最大编码位置
        :param max_position: 绝对位置编码最大位置数
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param type_vocab_size: segment_ids的词典大小
        :param hidden_dropout_prob: embedding、encoder和pool层中的全连接层dropout
        :param attention_probs_dropout_prob: attention的dropout
        :param shared_segment_embeddings: segment是否共享token embedding
        :param hierarchical_position: 是否层次分解位置编码
        :param initializer_range: truncated_normal_initializer初始化方法的stdev
        :param use_relative_position: 是否使用相对位置编码
        :param max_relative_positions: 相对位置编码最大位置数
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.max_position_embeddings = max_position_embeddings
        self.max_position = max_position or self.max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.shared_segment_embeddings = shared_segment_embeddings
        self.hierarchical_position = hierarchical_position
        self.initializer_range = initializer_range
        self.use_relative_position = use_relative_position
        self.max_relative_positions = max_relative_positions

    @classmethod
    def from_dict(cls, json_obj):
        """从字典对象中构建BertConfig
        :param json_obj: 字典对象
        :return: BertConfig
        """
        bert_config = BertConfig(**json_obj)
        for (key, value) in json_obj.items():
            if key == "relative_attention":
                bert_config.use_relative_position = value
            else:
                bert_config.__dict__[key] = value

        return bert_config

    @classmethod
    def from_json_file(cls, json_file_path: str):
        """从json文件中构建BertConfig
        :param json_file_path: JSON文件路径
        :return: BertConfig
        """
        with open(json_file_path, "r", encoding="utf-8") as reader:
            return cls.from_dict(json_obj=json.load(reader))

    def to_dict(self) -> dict:
        """将实例序列化为字典"""
        return copy.deepcopy(self.__dict__)

    def to_json_string(self) -> str:
        """将实例序列化为json字符串"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
    "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/config.json",
    "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/config.json",
    "bert-base-multilingual-uncased": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/config.json",
    "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json",
    "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/config.json",
    "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking": "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/config.json",
    "bert-large-cased-whole-word-masking": "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/config.json",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
    "bert-base-cased-finetuned-mrpc": "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
    "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/config.json",
    "bert-base-german-dbmdz-uncased": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char": "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/config.json",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-cased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json",
    "TurkuNLP/bert-base-finnish-uncased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json",
    "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    # See all BERT models at https://huggingface.co/models?filter=bert
}


class TBertConfig(PretrainedConfig):
    """用于给transformers的BertModel和TFBertModel使用的配置类
        默认配置使用https://huggingface.co/bert-base-uncased
    """
    model_type = "bert"

    def __init__(self,
                 vocab_size: int = 30522,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: Any = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 pad_token_id: int = 0,
                 gradient_checkpointing: bool = False,
                 position_embedding_type: str = "absolute",
                 segment_type: str = "absolute",
                 use_mean_pooling: bool = False,
                 use_cache: bool = True,
                 **kwargs):
        """构建TBertConfig
        :param vocab_size: 词表大小
        :param hidden_size: encoder和pool维度大小
        :param num_hidden_layers: encoder的层数
        :param num_attention_heads: encoder中的attention层的注意力头数量
        :param hidden_act: encoder和pool中的非线性激活函数
        :param intermediate_size: 前馈神经网络层维度大小
        :param hidden_dropout_prob: embedding、encoder和pool层中的全连接层dropout
        :param attention_probs_dropout_prob: attention的dropout
        :param max_position_embeddings: 绝对位置编码最大位置数
        :param type_vocab_size: token_type_ids的词典大小
        :param initializer_range: truncated_normal_initializer初始化方法的stdev
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param gradient_checkpointing: 是否移除模型中部分梯度，用于节约显存
        :param position_embedding_type: 位置编码类型，相对/绝对
        :param segment_type: segment相对位置还是绝对位置
        :param use_cache: 是否缓存
        :param use_mean_pooling: 是否增加pool输出层
        """
        super(TBertConfig, self).__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.segment_type = segment_type
        self.use_cache = use_cache
        self.use_mean_pooling = use_mean_pooling


class NeZhaConfig(PretrainedConfig):
    """用于NeZha模型使用的配置类
    """
    model_type = "bert"

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 max_relative_position=64,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 gradient_checkpointing=False,
                 position_embedding_type="absolute",
                 use_relative_position=True,
                 segment_type="absolute",
                 use_mean_pooling=False,
                 use_cache=True,
                 **kwargs):
        """构建TBertConfig
        :param vocab_size: 词表大小
        :param hidden_size: encoder和pool维度大小
        :param num_hidden_layers: encoder的层数
        :param num_attention_heads: encoder中的attention层的注意力头数量
        :param intermediate_size: 前馈神经网络层维度大小
        :param hidden_act: encoder和pool中的非线性激活函数
        :param hidden_dropout_prob: embedding、encoder和pool层中的全连接层dropout
        :param attention_probs_dropout_prob: attention的dropout
        :param max_position_embeddings: 绝对位置编码最大位置数
        :param max_relative_position: 相对位置编码最大位置数
        :param type_vocab_size: token_type_ids的词典大小
        :param initializer_range: truncated_normal_initializer初始化方法的stdev
        :param layer_norm_eps: layer norm 附加因子，避免除零
        :param pad_token_id: 用于padding的token id
        :param gradient_checkpointing: 是否移除模型中部分梯度，用于节约显存
        :param position_embedding_type: 位置编码类型，相对/绝对
        :param use_relative_position: 是否使用相对位置编码
        :param segment_type: segment相对位置还是绝对位置
        :param use_mean_pooling: 是否增加pool输出层
        :param use_cache: 是否缓存
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.segment_type = segment_type
        self.use_cache = use_cache
        self.use_mean_pooling = use_mean_pooling
        self.use_relative_position = use_relative_position
        self.max_relative_position = max_relative_position


def actuator() -> NoReturn:
    pass
