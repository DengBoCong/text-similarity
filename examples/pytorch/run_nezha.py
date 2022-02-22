#! -*- coding: utf-8 -*-
""" Pytorch Run NEZHA
"""
# Author: DengBoCong <bocongdeng@gmail.com>
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
from sim.pytorch import bert_variable_mapping
from sim.pytorch.common import Checkpoint
from sim.pytorch.common import load_bert_weights
from sim.pytorch.common import set_seed
from sim.pytorch.common import truncated_normal_
from sim.pytorch.modeling_nezha import NEZHA
from sim.pytorch.pipeline import TextPairPipeline
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import NormalDataGenerator
from sim.tools.data_processor.process_plain_text import text_to_token_id_for_bert
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


class Model(nn.Module):
    """组合模型适配任务
    """

    def __init__(self, bert_config: BertConfig, batch_size: int):
        super(Model, self).__init__()
        self.nezha = NEZHA(config=bert_config, batch_size=batch_size, with_pool=True)
        self.class_dropout = nn.Dropout(p=0.1)
        self.class_dense = nn.Linear(in_features=bert_config.hidden_size, out_features=2)
        truncated_normal_(stddev=bert_config.initializer_range)(self.class_dense.weight)

    def forward(self, input_ids, token_type_ids):
        outputs = self.bert_model(input_ids, token_type_ids)
        outputs = self.class_dropout(outputs)
        outputs = self.class_dense(outputs)

        return outputs


def actuator(model_dir: str, execute_type: str) -> NoReturn:
    """
    :param model_dir: 预训练模型目录
    :param execute_type: 执行类型
    """
    pad_max_len = 40
    batch_size = 64
    seed = 1
    epochs = 5
    raw_train_data_path = "./corpus/chinese/LCQMC/train.txt"
    raw_valid_data_path = "./corpus/chinese/LCQMC/test.txt"
    train_data_path = "./data/train1.txt"
    valid_data_path = "./data/test1.txt"
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
                                  pad_max_len=pad_max_len, token_dict=dict_path)
        logger.info("Begin preprocess valid data")
        text_to_token_id_for_bert(file_path=raw_valid_data_path, save_path=valid_data_path,
                                  pad_max_len=pad_max_len, token_dict=dict_path)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = NormalDataGenerator(train_file.readlines(), batch_size)
            valid_generator = NormalDataGenerator(valid_file.readlines(), batch_size, random=False)

        bert_config = BertConfig.from_json_file(json_file_path=config_path)
        model = Model(bert_config=bert_config, batch_size=batch_size)

        weight_dict = load_bert_weights(
            model_file_path=model_file_path, model=model,
            mapping=bert_variable_mapping(bert_config.num_hidden_layers, prefix_="nezha.")
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        pipeline = TextPairPipeline([model], batch_size, device, torch.IntTensor, torch.LongTensor)
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
    actuator(model_dir="./data/ch/bert/nezha-base-wwm", execute_type="train")
