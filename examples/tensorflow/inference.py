#! -*- coding: utf-8 -*-
""" TensorFlow Inference
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow.keras as keras
from sim.tensorflow.common import load_weights_from_checkpoint
from sim.tensorflow.modeling_albert import albert
from sim.tensorflow.modeling_bert import bert_model
from sim.tensorflow.modeling_colbert import colbert
from sim.tensorflow.modeling_nezha import NEZHA
from sim.tensorflow.modeling_text_cnn import text_cnn
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import InferSample
from sim.tools.tokenizer import BertTokenizer


class Inference(object):
    """推断"""

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", 1)
        self.config_path = kwargs.get("config_path", "")  # bert/albert/nezha/cnn/colbert
        self.checkpoint_path = kwargs.get("checkpoint_path", "")  # bert/albert/nezha/cnn/colbert
        self.tokenizer = kwargs.get("tokenizer", None)  # bert/albert/nezha/cnn/colbert
        self.pad_max_len = kwargs.get("pad_max_len", 20)  # bert/albert/nezha/cnn/colbert
        self.variable_mapping = kwargs.get("variable_mapping", None)  # bert/albert/nezha/cnn/colbert
        self.units = kwargs.get("units", None)  # cnn
        self.filter_num = kwargs.get("filter_num", None)  # cnn
        self.kernel_sizes = kwargs.get("kernel_sizes", None)  # cnn
        self.initializers = kwargs.get("initializers", None)  # cnn
        self.activations = kwargs.get("activations", None)  # cnn
        self.padding = kwargs.get("padding", None)  # cnn
        self.model_type = kwargs.get("model_type", None)  # cnn/colbert

    def text_to_id_for_bert(self, pair: tuple, transfer: str = "normal"):
        """
        :param pair: 输入
        :param transfer: 文本处理形式
        """
        if self.tokenizer is None:
            raise ValueError("`tokenizer` must not be None")
        if transfer == "single":
            a_token_ids, a_segment_ids = self.tokenizer.encode(first_text=pair[0], max_len=self.pad_max_len)
            b_token_ids, b_segment_ids = self.tokenizer.encode(first_text=pair[1], max_len=self.pad_max_len)
            a_token_ids = a_token_ids + [0] * (self.pad_max_len - len(a_token_ids))
            a_segment_ids = a_segment_ids + [0] * (self.pad_max_len - len(a_segment_ids))
            b_token_ids = b_token_ids + [0] * (self.pad_max_len - len(b_token_ids))
            b_segment_ids = b_segment_ids + [0] * (self.pad_max_len - len(b_segment_ids))
            return a_token_ids, b_token_ids, a_segment_ids, b_segment_ids
        elif transfer == "normal":
            token_ids, segment_ids = self.tokenizer.encode(first_text=pair[0], second_text=pair[1],
                                                           max_len=self.pad_max_len)
            token_ids = token_ids + [0] * (self.pad_max_len - len(token_ids))
            segment_ids = segment_ids + [0] * (self.pad_max_len - len(segment_ids))
            return token_ids, segment_ids
        else:
            raise ValueError(f"`{transfer}` must be single or normal")

    def infer_basic_bert(self, sample: InferSample):
        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)

        bert = bert_model(config=bert_config, batch_size=self.batch_size, with_pool=True)
        outputs = keras.layers.Dropout(rate=0.1)(bert.output)
        outputs = keras.layers.Dense(
            units=2, activation="softmax",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)
        )(outputs)
        model = keras.Model(inputs=bert.inputs, outputs=outputs)
        load_weights_from_checkpoint(self.checkpoint_path, model, self.variable_mapping)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        outputs = model(inputs=[np.asarray([input_ids]), np.asarray([segment_ids])])

        return outputs[0][1]

    def infer_albert(self, sample: InferSample):
        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)

        albert_model = albert(config=bert_config, batch_size=self.batch_size)
        outputs = keras.layers.Lambda(lambda x: x[:, 0], name="cls-token")(albert_model.output)
        outputs = keras.layers.Dense(
            units=2, activation="softmax",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)
        )(outputs)
        model = keras.Model(inputs=albert_model.inputs, outputs=outputs)
        load_weights_from_checkpoint(self.checkpoint_path, albert_model, self.variable_mapping)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        outputs = model(inputs=[np.asarray([input_ids]), np.asarray([segment_ids])])

        return outputs[0][1]

    def infer_nezha(self, sample: InferSample):
        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)

        nezha = NEZHA(config=bert_config, batch_size=self.batch_size, with_pool=True)
        outputs = keras.layers.Dropout(rate=0.1)(nezha.output)
        outputs = keras.layers.Dense(
            units=2, activation="softmax",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range),
        )(outputs)
        model = keras.Model(inputs=nezha.inputs, outputs=outputs)
        load_weights_from_checkpoint(self.checkpoint_path, nezha, self.variable_mapping)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        outputs = model(inputs=[np.asarray([input_ids]), np.asarray([segment_ids])])

        return outputs[0][1]

    def infer_cnn_base(self, sample: InferSample):
        # 这里使用bert作为Embedding
        if self.model_type == "bert":
            bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
            bert = bert_model(config=bert_config, batch_size=self.batch_size)
        elif self.model_type == "albert":
            bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
            bert = albert(config=bert_config, batch_size=self.batch_size)
        elif self.model_type == "nezha":
            bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
            bert = NEZHA(config=bert_config, batch_size=self.batch_size)
        else:
            raise ValueError("`model_type` must in bert/albert/nezha")

        outputs = text_cnn(self.pad_max_len, bert_config.hidden_size, self.units, self.filter_num,
                           self.kernel_sizes, self.initializers, self.activations, self.padding)(bert.outputs)
        model = keras.Model(inputs=bert.inputs, outputs=outputs)
        load_weights_from_checkpoint(self.checkpoint_path, model, self.variable_mapping)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        outputs = model(inputs=[np.asarray([input_ids]), np.asarray([segment_ids])])

        return outputs[0][1]

    def infer_colbert(self, sample: InferSample):
        # 这里使用bert作为Embedding
        if self.model_type == "bert":
            bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
            bert = bert_model(config=bert_config, batch_size=self.batch_size)
        elif self.model_type == "albert":
            bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
            bert = albert(config=bert_config, batch_size=self.batch_size)
        elif self.model_type == "nezha":
            bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
            bert = NEZHA(config=bert_config, batch_size=self.batch_size)
        else:
            raise ValueError("`model_type` must in bert/albert/nezha")

        model = colbert(encoder_model=bert, mask_punctuation=False, tokenizer=self.tokenizer)
        load_weights_from_checkpoint(self.checkpoint_path, model, self.variable_mapping)

        a_token_ids, b_token_ids, a_segment_ids, b_segment_ids = self.text_to_id_for_bert((sample.get(0),
                                                                                           sample.get(1)), "single")
        outputs = model(inputs=[np.asarray([a_token_ids]), np.asarray([a_segment_ids]),
                                np.asarray([b_token_ids]), np.asarray([b_segment_ids])])

        return outputs


if __name__ == '__main__':
    basic_model_dir = "./data/ch/bert/basic_train_step_100000"
    config_path = os.path.join(basic_model_dir, "bert_config.json")
    checkpoint_path = os.path.join(basic_model_dir, "bert_model.ckpt")
    token_dict = os.path.join(basic_model_dir, "vocab.txt")

    inferSample = InferSample(0, ("七夕情人节送什么给女朋友好呢", "七夕情人节送女朋友什么好"))
    inference = Inference(config_path=config_path, checkpoint_path=checkpoint_path,
                          tokenizer=BertTokenizer(token_dict=token_dict, do_lower_case=True),
                          pad_max_len=40)
    print(inference.infer_basic_bert(inferSample))
