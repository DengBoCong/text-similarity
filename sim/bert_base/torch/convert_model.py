#! -*- coding: utf-8 -*-
""" Tensorflow Version Actuator
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import torch
from collections import OrderedDict
from simpletransformers_a
from transformers import BertForMaskedLM
from transformers import BertTokenizer
from typing import NoReturn


def convert_uer_to_transformers(input_model_path: str, output_model_path: str,
                                layers_num: int, target: str = "bert") -> NoReturn:
    """ 将uer框架的模型转换成transformers使用
    :param input_model_path: uer模型存放路径，mixed_corpus_bert_base_model.bin
    :param output_model_path: transformers模型存放路径，bert_base.bin
    :param layers_num: 层数
    :param target: 模型类型
    :return: None
    """
    input_model = torch.load(input_model_path)
    output_model = OrderedDict()

    output_model["bert.embeddings.word_embeddings.weight"] = input_model["embedding.word_embedding.weight"]
    output_model["bert.embeddings.position_embeddings.weight"] = input_model["embedding.position_embedding.weight"]
    output_model["bert.embeddings.token_type_embeddings.weight"] = input_model["embedding.segment_embedding.weight"][1:, :]
    output_model["bert.embeddings.LayerNorm.weight"] = input_model["embedding.layer_norm.gamma"]
    output_model["bert.embeddings.LayerNorm.bias"] = input_model["embedding.layer_norm.beta"]

    for i in range(layers_num):
        output_model[f"bert.encoder.layer.{i}.attention.self.query.weight"] = input_model[f"encoder.transformer.{i}.self_attn.linear_layers.0.weight"]
        output_model[f"bert.encoder.layer.{i}.attention.self.query.bias"] = input_model[f"encoder.transformer.{i}.self_attn.linear_layers.0.bias"]
        output_model[f"bert.encoder.layer.{i}.attention.self.key.weight"] = input_model[f"encoder.transformer.{i}.self_attn.linear_layers.1.weight"]
        output_model[f"bert.encoder.layer.{i}.attention.self.key.bias"] = input_model[f"encoder.transformer.{i}.self_attn.linear_layers.1.bias"]
        output_model[f"bert.encoder.layer.{i}.attention.self.value.weight"] = input_model[f"encoder.transformer.{i}.self_attn.linear_layers.2.weight"]
        output_model[f"bert.encoder.layer.{i}.attention.self.value.bias"] = input_model[f"encoder.transformer.{i}.self_attn.linear_layers.2.bias"]
        output_model[f"bert.encoder.layer.{i}.attention.output.dense.weight"] = input_model[f"encoder.transformer.{i}.self_attn.final_linear.weight"]
        output_model[f"bert.encoder.layer.{i}.attention.output.dense.bias"] = input_model[f"encoder.transformer.{i}.self_attn.final_linear.bias"]
        output_model[f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight"] = input_model[f"encoder.transformer.{i}.layer_norm_1.gamma"]
        output_model[f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias"] = input_model[f"encoder.transformer.{i}.layer_norm_1.beta"]
        output_model[f"bert.encoder.layer.{i}.intermediate.dense.weight"] = input_model[f"encoder.transformer.{i}.feed_forward.linear_1.weight"]
        output_model[f"bert.encoder.layer.{i}.intermediate.dense.bias"] = input_model[f"encoder.transformer.{i}.feed_forward.linear_1.bias"]
        output_model[f"bert.encoder.layer.{i}.output.dense.weight"] = input_model[f"encoder.transformer.{i}.feed_forward.linear_2.weight"]
        output_model[f"bert.encoder.layer.{i}.output.dense.bias"] = input_model[f"encoder.transformer.{i}.feed_forward.linear_2.bias"]
        output_model[f"bert.encoder.layer.{i}.output.LayerNorm.weight"] = input_model[f"encoder.transformer.{i}.layer_norm_2.gamma"]
        output_model[f"bert.encoder.layer.{i}.output.LayerNorm.bias"] = input_model[f"encoder.transformer.{i}.layer_norm_2.beta"]

    if target == "bert":
        output_model["bert.pooler.dense.weight"] = input_model["target.nsp_linear_1.weight"]
        output_model["bert.pooler.dense.bias"] = input_model["target.nsp_linear_1.bias"]
        output_model["cls.seq_relationship.weight"] = input_model["target.nsp_linear_2.weight"]
        output_model["cls.seq_relationship.bias"] = input_model["target.nsp_linear_2.bias"]

    output_model["cls.predictions.transform.dense.weight"] = input_model["target.mlm_linear_1.weight"]
    output_model["cls.predictions.transform.dense.bias"] = input_model["target.mlm_linear_1.bias"]
    output_model["cls.predictions.transform.LayerNorm.weight"] = input_model["target.layer_norm.gamma"]
    output_model["cls.predictions.transform.LayerNorm.bias"] = input_model["target.layer_norm.beta"]
    output_model["cls.predictions.decoder.weight"] = input_model["target.mlm_linear_2.weight"]
    output_model["cls.predictions.bias"] = input_model["target.mlm_linear_2.bias"]

    torch.save(output_model, output_model_path)


def convert_model_to_transformers(input_model_name: str, output_model_name: str,
                                  model_type: str, tokenizer_model: str) -> NoReturn:
    """ 将指定模型转化为transformers使用
    :param input_model_name: 指定模型所在目录/在线模型名
    :param output_model_name: 保存模型目录/模型名
    :param model_type: 模型类型
    :param tokenizer_model:
    :return: None
    """


