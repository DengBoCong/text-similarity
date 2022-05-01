#! -*- coding: utf-8 -*-
""" Pytorch Inference
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import torch
import torch.nn as nn
from examples.pytorch.run_albert import Model as AlBert
from examples.pytorch.run_basic_bert import Model as BasicBertModel
from examples.pytorch.run_cnn_base import Model as TextCNN
from examples.pytorch.run_nezha import Model as NEZHA
from sim.pytorch import bert_variable_mapping
from sim.pytorch.common import load_weights
from sim.pytorch.modeling_colbert import ColBERT
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import InferSample
from sim.tools.tokenizer import BertTokenizer


class Inference(object):
    """推断"""

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", 1)
        self.config_path = kwargs.get("config_path", "")
        self.tokenizer = kwargs.get("tokenizer", None)
        self.pad_max_len = kwargs.get("pad_max_len", 20)
        self.model_file_path = kwargs.get("model_file_path", "")
        self.variable_mapping = kwargs.get("variable_mapping", None)
        self.inp_dtype = kwargs.get("inp_dtype", torch.int32)
        self.model_type = kwargs.get("model_type", None)

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
        model = BasicBertModel(bert_config=bert_config, batch_size=self.batch_size)

        weight_dict = load_weights(
            model_file_path=self.model_file_path,
            model=model, mapping=self.variable_mapping
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        inputs1 = torch.from_numpy(np.asarray([input_ids])).type(self.inp_dtype).to(device)
        inputs2 = torch.from_numpy(np.asarray([segment_ids])).type(self.inp_dtype).to(device)
        outputs = model(inputs1, inputs2)
        outputs = torch.softmax(outputs, dim=-1)

        return outputs[0][1]

    def infer_albert(self, sample: InferSample):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
        model = AlBert(bert_config=bert_config, batch_size=self.batch_size)

        weight_dict = load_weights(
            model_file_path=self.model_file_path, model=model, mapping=self.variable_mapping
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        inputs1 = torch.from_numpy(np.asarray([input_ids])).type(self.inp_dtype).to(device)
        inputs2 = torch.from_numpy(np.asarray([segment_ids])).type(self.inp_dtype).to(device)
        outputs = model(inputs1, inputs2)
        outputs = torch.softmax(outputs, dim=-1)

        return outputs[0][1]

    def infer_nezha(self, sample: InferSample):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
        model = NEZHA(bert_config=bert_config, batch_size=self.batch_size)

        weight_dict = load_weights(
            model_file_path=self.model_file_path, model=model, mapping=self.variable_mapping
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        inputs1 = torch.from_numpy(np.asarray([input_ids])).type(self.inp_dtype).to(device)
        inputs2 = torch.from_numpy(np.asarray([segment_ids])).type(self.inp_dtype).to(device)
        outputs = model(inputs1, inputs2)
        outputs = torch.softmax(outputs, dim=-1)

        return outputs[0][1]

    def infer_cnn_base(self, sample: InferSample):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
        model = TextCNN(bert_config=bert_config, batch_size=self.batch_size, seq_len=self.pad_max_len, filter_num=300)

        weight_dict = load_weights(
            model_file_path=self.model_file_path, model=model, mapping=self.variable_mapping
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        input_ids, segment_ids = self.text_to_id_for_bert((sample.get(0), sample.get(1)))
        inputs1 = torch.from_numpy(np.asarray([input_ids])).type(self.inp_dtype).to(device)
        inputs2 = torch.from_numpy(np.asarray([segment_ids])).type(self.inp_dtype).to(device)
        outputs = model(inputs1, inputs2)
        outputs = torch.softmax(outputs, dim=-1)

        return outputs[0][1]

    def infer_colbert(self, sample: InferSample):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bert_config = BertConfig.from_json_file(json_file_path=self.config_path)
        model = ColBERT(config=bert_config, batch_size=self.batch_size, bert_model_type=self.model_type,
                        mask_punctuation=False, tokenizer=self.tokenizer)

        weight_dict = load_weights(
            model_file_path=self.model_file_path, model=model, mapping=self.variable_mapping
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        a_token_ids, b_token_ids, a_segment_ids, b_segment_ids = self.text_to_id_for_bert((sample.get(0),
                                                                                           sample.get(1)), "single")
        inputs1 = torch.from_numpy(np.asarray([a_token_ids])).type(self.inp_dtype).to(device)
        inputs2 = torch.from_numpy(np.asarray([a_segment_ids])).type(self.inp_dtype).to(device)
        inputs3 = torch.from_numpy(np.asarray([b_token_ids])).type(self.inp_dtype).to(device)
        inputs4 = torch.from_numpy(np.asarray([b_segment_ids])).type(self.inp_dtype).to(device)
        outputs = model(inputs1, inputs2, inputs3, inputs4)

        return outputs


if __name__ == '__main__':
    basic_model_dir = "./data/chinese_bert_pytorch"
    config_path = os.path.join(basic_model_dir, "bert_config.json")
    model_file_path = os.path.join(basic_model_dir, "pytorch_model.bin")
    dict_path = os.path.join(basic_model_dir, "vocab.txt")

    inferSample = InferSample(0, ("七夕情人节送什么给女朋友好呢", "七夕情人节送女朋友什么好"))
    inference = Inference(config_path=config_path, model_file_path=model_file_path,
                          tokenizer=BertTokenizer(token_dict=dict_path, do_lower_case=True),
                          pad_max_len=40, variable_mapping=bert_variable_mapping(12, prefix_="bert_model."),
                          inp_dtype=torch.IntTensor)

    print(inference.infer_basic_bert(inferSample))
