#! -*- coding: utf-8 -*-
""" Some TensorFlow Components
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def bert_variable_mapping(num_hidden_layers: int):
    """映射到官方BERT权重格式
    :param num_hidden_layers: encoder的层数
    """
    mapping = {
        "embedding-token/embeddings": "bert/embeddings/word_embeddings",
        "embedding-segment/embeddings": "bert/embeddings/token_type_embeddings",
        "embedding-position/embeddings": "bert/embeddings/position_embeddings",
        "embedding-norm/gamma": "bert/embeddings/LayerNorm/gamma",
        "embedding-norm/beta": "bert/embeddings/LayerNorm/beta",
        "embedding-mapping/kernel": "bert/encoder/embedding_hidden_mapping_in/kernel",
        "embedding-mapping/bias": "bert/encoder/embedding_hidden_mapping_in/bias",
        "bert-output/pooler-dense/kernel": "bert/pooler/dense/kernel",
        "bert-output/pooler-dense/bias": "bert/pooler/dense/bias",
        "bert-output/nsp-prob/kernel": "cls/seq_relationship/output_weights",
        "bert-output/nsp-prob/bias": "cls/seq_relationship/output_bias",
        "bert-output/mlm-dense/kernel": "cls/predictions/transform/dense/kernel",
        "bert-output/mlm-dense/bias": "cls/predictions/transform/dense/bias",
        "bert-output/mlm-norm/gamma": "cls/predictions/transform/LayerNorm/gamma",
        "bert-output/mlm-norm/beta": "cls/predictions/transform/LayerNorm/beta",
        "bert-output/mlm-bias/bias": "cls/predictions/output_bias"
    }

    for i in range(num_hidden_layers):
        prefix = 'bert/encoder/layer_%d/' % i
        mapping.update({
            f"bert-layer-{i}/multi-head-self-attention/query/kernel": prefix + "attention/self/query/kernel",
            f"bert-layer-{i}/multi-head-self-attention/query/bias": prefix + "attention/self/query/bias",
            f"bert-layer-{i}/multi-head-self-attention/key/kernel": prefix + "attention/self/key/kernel",
            f"bert-layer-{i}/multi-head-self-attention/key/bias": prefix + "attention/self/key/bias",
            f"bert-layer-{i}/multi-head-self-attention/value/kernel": prefix + "attention/self/value/kernel",
            f"bert-layer-{i}/multi-head-self-attention/value/bias": prefix + "attention/self/value/bias",
            f"bert-layer-{i}/multi-head-self-attention/output/kernel": prefix + "attention/output/dense/kernel",
            f"bert-layer-{i}/multi-head-self-attention/output/bias": prefix + "attention/output/dense/bias",
            f"bert-layer-{i}/multi-head-self-attention-norm/gamma": prefix + "attention/output/LayerNorm/gamma",
            f"bert-layer-{i}/multi-head-self-attention-norm/beta": prefix + "attention/output/LayerNorm/beta",
            f"bert-layer-{i}/feedforward/input/kernel": prefix + "intermediate/dense/kernel",
            f"bert-layer-{i}/feedforward/input/bias": prefix + "intermediate/dense/bias",
            f"bert-layer-{i}/feedforward/output/kernel": prefix + "output/dense/kernel",
            f"bert-layer-{i}/feedforward/output/bias": prefix + "output/dense/bias",
            f"bert-layer-{i}/feedforward-norm/gamma": prefix + "output/LayerNorm/gamma",
            f"bert-layer-{i}/feedforward-norm/beta": prefix + "output/LayerNorm/beta",
        })

    return mapping


def albert_variable_mapping():
    """映射到官方ALBERT权重格式
    """
    mapping = {
        "embedding-token/embeddings": "bert/embeddings/word_embeddings",
        "embedding-segment/embeddings": "bert/embeddings/token_type_embeddings",
        "embedding-position/embeddings": "bert/embeddings/position_embeddings",
        "embedding-norm/gamma": "bert/embeddings/LayerNorm/gamma",
        "embedding-norm/beta": "bert/embeddings/LayerNorm/beta",
        "embedding-mapping/kernel": "bert/encoder/embedding_hidden_mapping_in/kernel",
        "embedding-mapping/bias": "bert/encoder/embedding_hidden_mapping_in/bias",
        "bert-output/pooler-dense/kernel": "bert/pooler/dense/kernel",
        "bert-output/pooler-dense/bias": "bert/pooler/dense/bias",
        "bert-output/nsp-prob/kernel": "cls/seq_relationship/output_weights",
        "bert-output/nsp-prob/bias": "cls/seq_relationship/output_bias",
        "bert-output/mlm-dense/kernel": "cls/predictions/transform/dense/kernel",
        "bert-output/mlm-dense/bias": "cls/predictions/transform/dense/bias",
        "bert-output/mlm-norm/gamma": "cls/predictions/transform/LayerNorm/gamma",
        "bert-output/mlm-norm/beta": "cls/predictions/transform/LayerNorm/beta",
        "bert-output/mlm-bias/bias": "cls/predictions/output_bias"
    }

    prefix = "bert/encoder/transformer/group_0/inner_group_0/"
    mapping.update({
        "bert-layer/multi-head-self-attention/query/kernel": prefix + "attention_1/self/query/kernel",
        "bert-layer/multi-head-self-attention/query/bias": prefix + "attention_1/self/query/bias",
        "bert-layer/multi-head-self-attention/key/kernel": prefix + "attention_1/self/key/kernel",
        "bert-layer/multi-head-self-attention/key/bias": prefix + "attention_1/self/key/bias",
        "bert-layer/multi-head-self-attention/value/kernel": prefix + "attention_1/self/value/kernel",
        "bert-layer/multi-head-self-attention/value/bias": prefix + "attention_1/self/value/bias",
        "bert-layer/multi-head-self-attention/output/kernel": prefix + "attention_1/output/dense/kernel",
        "bert-layer/multi-head-self-attention/output/bias": prefix + "attention_1/output/dense/bias",
        "bert-layer/multi-head-self-attention-norm/gamma": prefix + "LayerNorm/gamma",
        "bert-layer/multi-head-self-attention-norm/beta": prefix + "LayerNorm/beta",
        "bert-layer/feedforward/input/kernel": prefix + "ffn_1/intermediate/dense/kernel",
        "bert-layer/feedforward/input/bias": prefix + "ffn_1/intermediate/dense/bias",
        "bert-layer/feedforward/output/kernel": prefix + "ffn_1/intermediate/output/dense/kernel",
        "bert-layer/feedforward/output/bias": prefix + "ffn_1/intermediate/output/dense/bias",
        "bert-layer/feedforward-norm/gamma": prefix + "LayerNorm_1/gamma",
        "bert-layer/feedforward-norm/beta": prefix + "LayerNorm_1/beta",
    })

    return mapping


def deberta_variable_mapping():
    """映射到官方DeBERTa权重格式
    """
    pass
