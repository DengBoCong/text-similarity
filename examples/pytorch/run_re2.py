#! -*- coding: utf-8 -*-
""" Pytorch Run Simple-Effective-Text-Matching RE2
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import torch
import torch.nn as nn
from datetime import datetime
from sim.pytorch.modeling_re2 import Re2Network
from sim.tools.data_processor.data_format import NormalDataGenerator
from sim.tools.data_processor.process_plain_text import text_pair_to_token_id
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from sim.pytorch.common import Checkpoint
from sim.pytorch.common import set_seed
from sim.pytorch.pipeline import TextPairPipeline
from typing import Any
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


class CustomPipeline(TextPairPipeline):
    def __init__(self, model: list, batch_size: int, device: Any, inp_dtype: Any, lab_dtype: Any):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        :param device: 设备
        :param inp_dtype: 输入类型
        :param lab_dtype: 标签类型
        """
        super(CustomPipeline, self).__init__(model, batch_size, device, inp_dtype, lab_dtype)

    def _metrics(self, y_true: Any, y_pred: Any):
        """指标计算
        :param y_true: 真实标签
        :param y_pred: 预测值
        """
        y_pred = torch.softmax(input=y_pred, dim=-1)
        loss = nn.CrossEntropyLoss()(y_pred, y_true)
        accuracy = torch.eq(torch.argmax(y_pred, dim=-1), y_true).sum(dim=-1).div(self.batch_size)

        return loss, accuracy


def actuator(config_path: str, execute_type: str) -> NoReturn:
    """
    :param config_path: 配置json文件路径
    :param execute_type: 执行类型
    """
    with open(config_path, "r", encoding="utf-8") as file:
        options = json.load(file)

    # 这里在日志文件里面做一个执行分割
    key = str(datetime.now())
    logger.info("========================{}========================".format(key))
    # 训练时保存模型配置
    if execute_type == "train" and not save_model_config(key=key, model_desc="RNN Base",
                                                         model_config=options, config_path=MODEL_CONFIG_FILE_PATH):
        raise EOFError("An error occurred while saving the configuration file")

    if execute_type == "preprocess":
        logger.info("Begin preprocess train data")
        tokenizer = text_pair_to_token_id(file_path=options["raw_train_data_path"],
                                          save_path=options["train_data_path"], pad_max_len=options["seq_len"])
        logger.info("Begin preprocess valid data")
        text_pair_to_token_id(file_path=options["raw_valid_data_path"], save_path=options["valid_data_path"],
                              pad_max_len=options["seq_len"], tokenizer=tokenizer)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(options["train_data_path"], "r", encoding="utf-8") as train_file, open(
                options["valid_data_path"], "r", encoding="utf-8") as valid_file:
            train_generator = NormalDataGenerator(train_file.readlines(), options["batch_size"])
            valid_generator = NormalDataGenerator(valid_file.readlines(), options["batch_size"])

        model = Re2Network(vocab_size=options["vocab_size"], embedding_size=options["embedding_dim"],
                           block_layer_num=options["block_layer_num"], enc_layers=options["enc_layers"],
                           enc_kernel_size=options["enc_kernel_size"], dropout=options["dropout"],
                           hidden_size=options["hidden_size"], connection_args=options["connection_args"],
                           align_type=options["align_type"], fusion_type=options["fusion_type"],
                           pred_type=options["pred_type"])

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        pipeline = CustomPipeline([model], options["batch_size"], device, torch.IntTensor, torch.LongTensor)
        history = {"t_acc": [], "t_loss": [], "v_acc": [], "v_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=options["seed"])
            optimizer = torch.optim.Adam(params=model.parameters())
            checkpoint = Checkpoint(checkpoint_dir=options["checkpoint_dir"], optimizer=optimizer, model=model)

            pipeline.train(train_generator, valid_generator, options["epochs"], optimizer,
                           checkpoint, options["checkpoint_save_freq"], history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(config_path="./data/config/re2.json", execute_type="preprocess")
