#! -*- coding: utf-8 -*-
""" Some Pytorch Components
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def bert_variable_mapping(num_hidden_layers: int, prefix_: str = ""):
    """映射到官方BERT权重格式
    :param num_hidden_layers: encoder的层数
    :param prefix_: 如果是对项目内的bert组件进行组合用以适配任务，那么在加载
                    标准bert权重的时候，由于pytorch权重命名规则，可能会需要prefix
    """
    mapping = {
        prefix_ + "token_embeddings.weight": "bert.embeddings.word_embeddings.weight",
        prefix_ + "bert_embeddings.segment_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
        prefix_ + "bert_embeddings.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
        prefix_ + "bert_embeddings.layer_norm.weight": "bert.embeddings.LayerNorm.weight",
        prefix_ + "bert_embeddings.layer_norm.bias": "bert.embeddings.LayerNorm.bias",
        prefix_ + "bert_output.pooler_dense.weight": "bert.pooler.dense.weight",
        prefix_ + "bert_output.pooler_dense.bias": "bert.pooler.dense.bias",
        prefix_ + "bert_output.nsp_prob.weight": "cls.seq_relationship.weight",
        prefix_ + "bert_output.nsp_prob.bias": "cls.seq_relationship.bias",
        prefix_ + "bert_output.mlm_decoder.weight": "cls.predictions.decoder.weight",
        prefix_ + "bert_output.mlm_dense.weight": "cls.predictions.transform.dense.weight",
        prefix_ + "bert_output.mlm_dense.bias": "cls.predictions.transform.dense.bias",
        prefix_ + "bert_output.mlm_norm.weight": "cls.predictions.transform.LayerNorm.weight",
        prefix_ + "bert_output.mlm_norm.bias": "cls.predictions.transform.LayerNorm.bias",
        prefix_ + "bert_output.mlm_bias.weight": "cls.predictions.bias"
    }

    for i in range(num_hidden_layers):
        prefix = f"bert.encoder.layer.{i}."
        mapping.update({
            f"{prefix_}bert_layer_{i}.bert_self_attention.query_dense.weight": prefix + "attention.self.query.weight",
            f"{prefix_}bert_layer_{i}.bert_self_attention.query_dense.bias": prefix + "attention.self.query.bias",
            f"{prefix_}bert_layer_{i}.bert_self_attention.key_dense.weight": prefix + "attention.self.key.weight",
            f"{prefix_}bert_layer_{i}.bert_self_attention.key_dense.bias": prefix + "attention.self.key.bias",
            f"{prefix_}bert_layer_{i}.bert_self_attention.value_dense.weight": prefix + "attention.self.value.weight",
            f"{prefix_}bert_layer_{i}.bert_self_attention.value_dense.bias": prefix + "attention.self.value.bias",
            f"{prefix_}bert_layer_{i}.bert_self_attention.output_dense.weight": prefix + "attention.output.dense.weight",
            f"{prefix_}bert_layer_{i}.bert_self_attention.output_dense.bias": prefix + "attention.output.dense.bias",
            f"{prefix_}bert_layer_{i}.attn_norm.weight": prefix + "attention.output.LayerNorm.weight",
            f"{prefix_}bert_layer_{i}.attn_norm.bias": prefix + "attention.output.LayerNorm.bias",
            f"{prefix_}bert_layer_{i}.feedforward.input_dense.weight": prefix + "intermediate.dense.weight",
            f"{prefix_}bert_layer_{i}.feedforward.input_dense.bias": prefix + "intermediate.dense.bias",
            f"{prefix_}bert_layer_{i}.feedforward.output_dense.weight": prefix + "output.dense.weight",
            f"{prefix_}bert_layer_{i}.feedforward.output_dense.bias": prefix + "output.dense.bias",
            f"{prefix_}bert_layer_{i}.feedforward_norm.weight": prefix + "output.LayerNorm.weight",
            f"{prefix_}bert_layer_{i}.feedforward_norm.bias": prefix + "output.LayerNorm.bias"
        })

    return mapping


def albert_variable_mapping(prefix_: str = ""):
    """映射到官方ALBERT权重格式
    :param prefix_: 如果是对项目内的bert组件进行组合用以适配任务，那么在加载
                    标准bert权重的时候，由于pytorch权重命名规则，可能会需要prefix
    """
    mapping = {
        prefix_ + "token_embeddings.weight": "albert.embeddings.word_embeddings.weight",
        prefix_ + "bert_embeddings.segment_embeddings.weight": "albert.embeddings.token_type_embeddings.weight",
        prefix_ + "bert_embeddings.position_embeddings.weight": "albert.embeddings.position_embeddings.weight",
        prefix_ + "bert_embeddings.layer_norm.weight": "albert.embeddings.LayerNorm.weight",
        prefix_ + "bert_embeddings.layer_norm.bias": "albert.embeddings.LayerNorm.bias",
        prefix_ + "bert_embeddings.outputs_dense.weight": "albert.encoder.embedding_hidden_mapping_in.weight",
        prefix_ + "bert_embeddings.outputs_dense.bias": "albert.encoder.embedding_hidden_mapping_in.bias",
        prefix_ + "bert_output.pooler_dense.weight": "albert.pooler.weight",
        prefix_ + "bert_output.pooler_dense.bias": "albert.pooler.bias",
        prefix_ + "albert.bert_output.mlm_decoder.weight": "predictions.decoder.weight",
        prefix_ + "albert.bert_output.mlm_decoder.bias": "predictions.decoder.bias",
        prefix_ + "bert_output.mlm_dense.weight": "predictions.dense.weight",
        prefix_ + "bert_output.mlm_dense.bias": "predictions.dense.bias",
        prefix_ + "bert_output.mlm_norm.weight": "predictions.LayerNorm.weight",
        prefix_ + "bert_output.mlm_norm.bias": "predictions.LayerNorm.bias",
        prefix_ + "bert_output.mlm_bias.weight": "predictions.bias",
        prefix_ + "bert_layer.bert_self_attention.query_dense.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
        prefix_ + "bert_layer.bert_self_attention.query_dense.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias",
        prefix_ + "bert_layer.bert_self_attention.key_dense.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
        prefix_ + "bert_layer.bert_self_attention.key_dense.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias",
        prefix_ + "bert_layer.bert_self_attention.value_dense.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
        prefix_ + "bert_layer.bert_self_attention.value_dense.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias",
        prefix_ + "bert_layer.bert_self_attention.output_dense.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
        prefix_ + "bert_layer.bert_self_attention.output_dense.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias",
        prefix_ + "bert_layer.attn_norm.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight",
        prefix_ + "bert_layer.attn_norm.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias",
        prefix_ + "bert_layer.feedforward.input_dense.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",
        prefix_ + "bert_layer.feedforward.input_dense.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias",
        prefix_ + "bert_layer.feedforward.output_dense.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight",
        prefix_ + "bert_layer.feedforward.output_dense.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias",
        prefix_ + "bert_layer.feedforward_norm.weight": "albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight",
        prefix_ + "bert_layer.feedforward_norm.bias": "albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias",
    }

    return mapping
