#! -*- coding: utf-8 -*-
""" Pytorch Run SimCSE
"""
# Author: DengBoCong <bocongdeng@gmail.com>
# https://arxiv.org/pdf/2104.08821.pdf
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import torch
import torch.nn as nn
from datetime import datetime
from sim.pytorch import albert_variable_mapping
from sim.pytorch import bert_variable_mapping
from sim.pytorch.common import Checkpoint
from sim.pytorch.common import load_bert_weights
from sim.pytorch.common import set_seed
from sim.pytorch.modeling_albert import ALBERT
from sim.pytorch.modeling_bert import BertModel
from sim.pytorch.modeling_nezha import NEZHA
from sim.pytorch.pipeline import TextPairPipeline
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import SimCSEDataGenerator
from sim.tools.data_processor.process_plain_text import text_to_token_id_for_bert
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from typing import Any
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


class SimCSEPipeline(TextPairPipeline):
    def __init__(self, model: list, batch_size: int, device: Any, inp_dtype: Any, lab_dtype: Any):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        :param device: 设备
        :param inp_dtype: 输入类型
        :param lab_dtype: 标签类型
        """
        super(SimCSEPipeline, self).__init__(model, batch_size, device, inp_dtype, lab_dtype)

    def _metrics(self, y_true: Any, y_pred: Any):
        """指标计算, SimCSE
        :param y_true: 真实标签
        :param y_pred: 预测值
        """
        ids = torch.arange(0, y_pred.shape[0])
        ids_1 = ids[None, :]
        ids_2 = (ids + 1 - ids % 2 * 2)[:, None]
        y_true = torch.argmax(torch.eq(ids_1, ids_2).float(), dim=-1)

        y_pred = torch.norm(y_pred, p="fro", dim=1, keepdim=True)
        sim = torch.matmul(y_pred, y_pred.permute(1, 0))
        sim = (sim - torch.eye(y_pred.shape[0]) * 1e12) * 20

        loss = nn.CrossEntropyLoss()(sim, y_true)
        accuracy = torch.eq(torch.argmax(sim, dim=-1), y_true).sum(dim=-1).div(self.batch_size)

        return loss, accuracy


class Model(nn.Module):
    """组合模型适配任务
    """

    def __init__(self, bert_config: BertConfig, batch_size: int, model_type: str, pooling: str):
        super(Model, self).__init__()
        self.first_block_layer_output = None
        self.last_block_layer_output = None
        self.pooling = pooling

        with_pool = "linear" if pooling == "pooler" else False
        if model_type == "bert":
            self.model = BertModel(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
        elif model_type == "albert":
            self.model = ALBERT(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
        elif model_type == "nezha":
            self.model = NEZHA(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
        else:
            raise ValueError("model_type is None or not support")

        if model_type == "albert":
            for (name, module) in self.model.named_modules():
                if name == "bert_layer.feedforward_norm":
                    module.register_forward_hook(hook=self.first_block_hook)
                    module.register_forward_hook(hook=self.last_block_hook)
                    break
        else:
            for (name, module) in self.model.named_modules():
                if name == "bert_layer_0.feedforward_norm":
                    module.register_forward_hook(hook=self.first_block_hook)
                elif name == f"bert_layer_{bert_config.num_hidden_layers - 1}.feedforward_norm":
                    module.register_forward_hook(hook=self.last_block_hook)

    def forward(self, input_ids, token_type_ids):
        outputs = self.model(input_ids, token_type_ids)

        if self.pooling == 'first-last-avg':
            first_block_outputs = self.first_block_layer_output.mean(dim=1)
            last_block_outputs = self.last_block_layer_output.mean(dim=1)
            outputs = (first_block_outputs + last_block_outputs) / 2.
        elif self.pooling == "last-avg":
            outputs = self.last_block_layer_output.mean(dim=1)
        elif self.pooling == "cls":
            outputs = self.last_block_layer_output[:, 0]
        elif self.pooling == "pooler":
            # pooler直接用输出，这里pass标记一下存在
            pass
        else:
            raise ValueError("pooling is None or not support")

        return outputs

    def first_block_hook(self, module, fea_in, fea_out):
        self.first_block_layer_output = fea_out
        return None

    def last_block_hook(self, module, fea_in, fea_out):
        self.last_block_layer_output = fea_out
        return


def actuator(model_dir: str, execute_type: str, model_type: str, pooling: str = "cls") -> NoReturn:
    """
    :param model_dir: 预训练模型目录
    :param execute_type: 执行类型
    :param model_type: 使用模型类型，注意加载对应的权重
    :param pooling: 输出层，论文中cls效果最好，'first-last-avg', 'last-avg', 'cls', 'pooler'
    """
    pad_max_len = 40
    batch_size = 64
    seed = 1
    epochs = 5
    raw_train_data_path = "./corpus/chinese/LCQMC/train.txt"
    raw_valid_data_path = "./corpus/chinese/LCQMC/test.txt"
    train_data_path = "./data/train_single.txt"
    valid_data_path = "./data/test_single.txt"
    checkpoint_dir = "./data/checkpoint/"
    checkpoint_save_size = 5
    checkpoint_save_freq = 2

    config_path = os.path.join(model_dir, "bert_config.json")
    model_file_path = os.path.join(model_dir, "pytorch_model.bin")
    dict_path = os.path.join(model_dir, "vocab.txt")

    with open(config_path, "r", encoding="utf-8") as file:
        options = json.load(file)

    # 这里在日志文件里面做一个执行分割
    key = str(datetime.now())
    logger.info("========================{}========================".format(key))
    # 训练时保存模型配置
    if execute_type == "train" and not save_model_config(key=key, model_desc="Bert Base",
                                                         model_config=options, config_path=MODEL_CONFIG_FILE_PATH):
        raise EOFError("An error occurred while saving the configuration file")

    if execute_type == "preprocess":
        logger.info("Begin preprocess train data")
        text_to_token_id_for_bert(file_path=raw_train_data_path, save_path=train_data_path,
                                  pad_max_len=pad_max_len, token_dict=dict_path, is_single=True)
        logger.info("Begin preprocess valid data")
        text_to_token_id_for_bert(file_path=raw_valid_data_path, save_path=valid_data_path,
                                  pad_max_len=pad_max_len, token_dict=dict_path, is_single=True)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = SimCSEDataGenerator(train_file.readlines(), batch_size)
            valid_generator = SimCSEDataGenerator(valid_file.readlines(), batch_size, random=False)

        bert_config = BertConfig.from_json_file(json_file_path=config_path)
        model = Model(bert_config, batch_size, model_type, pooling)

        if model_type == "albert":
            mapping = albert_variable_mapping(prefix_="model.")
        else:
            mapping = bert_variable_mapping(bert_config.num_hidden_layers, prefix_="model.")

        weight_dict = load_bert_weights(model_file_path=model_file_path, model=model, mapping=mapping)
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        pipeline = SimCSEPipeline([model], batch_size, device, torch.IntTensor, torch.LongTensor)
        history = {"t_acc": [], "t_loss": [], "v_acc": [], "v_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=seed)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)
            checkpoint = Checkpoint(checkpoint_dir=checkpoint_dir, optimizer=optimizer, model=model)

            pipeline.train(train_generator, valid_generator, epochs, optimizer,
                           checkpoint, checkpoint_save_freq, history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        elif execute_type == "inference":
            pass
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(model_dir="./data/ch/bert/chinese_wwm_L-12_H-768_A-12", execute_type="train", model_type="bert")
