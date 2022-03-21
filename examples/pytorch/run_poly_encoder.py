#! -*- coding: utf-8 -*-
""" Pytorch Run Basic Bert
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
from sim.pytorch.modeling_poly_encoder import PolyEncoder
from sim.pytorch.pipeline import TextPairPipeline
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import TetradDataGenerator
from sim.tools.data_processor.process_plain_text import tetrad_text_to_token_id_for_bert
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
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

    def _train_step(self, batch_dataset: dict, optimizer: torch.optim.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        inputs1 = torch.from_numpy(batch_dataset["inputs1"]).type(self.inp_dtype).to(self.device)
        inputs2 = torch.from_numpy(batch_dataset["inputs2"]).type(self.inp_dtype).to(self.device)
        inputs3 = torch.from_numpy(batch_dataset["inputs3"]).type(self.inp_dtype).to(self.device)
        inputs4 = torch.from_numpy(batch_dataset["inputs4"]).type(self.inp_dtype).to(self.device)
        labels = torch.from_numpy(batch_dataset["labels"]).type(self.lab_dtype).to(self.device)
        outputs = self.model[0](inputs1, inputs2, inputs3, inputs4)
        loss, accuracy = self._metrics(labels, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"t_loss": loss, "t_acc": accuracy}

    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 验证步
        :param batch_dataset: 验证步的当前batch数据
        """
        inputs1 = torch.from_numpy(batch_dataset["inputs1"]).type(self.inp_dtype).to(self.device)
        inputs2 = torch.from_numpy(batch_dataset["inputs2"]).type(self.inp_dtype).to(self.device)
        inputs3 = torch.from_numpy(batch_dataset["inputs3"]).type(self.inp_dtype).to(self.device)
        inputs4 = torch.from_numpy(batch_dataset["inputs4"]).type(self.inp_dtype).to(self.device)
        labels = torch.from_numpy(batch_dataset["labels"]).type(self.lab_dtype).to(self.device)
        with torch.no_grad():
            outputs = self.model[0](inputs1, inputs2, inputs3, inputs4)
            loss, accuracy = self._metrics(labels, outputs)

        return {"v_loss": loss, "v_acc": accuracy}

    def _save_model(self, *args, **kwargs) -> NoReturn:
        pass


def actuator(execute_type: str, model_type: str, model_dir: str = None) -> NoReturn:
    """
    :param execute_type: 执行类型
    :param model_type: 模型执行类型
    :param model_dir: 预训练模型目录
    """
    pad_max_len = 40
    batch_size = 64
    seed = 1
    epochs = 5
    poly_type = "learnt"
    candi_agg_type = "cls"
    poly_m = 16
    raw_train_data_path = "./corpus/chinese/LCQMC/train.txt"
    raw_valid_data_path = "./corpus/chinese/LCQMC/test.txt"
    train_data_path = "./data/train_tetrad.txt"
    valid_data_path = "./data/test_tetrad.txt"
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
        tetrad_text_to_token_id_for_bert(file_path=raw_train_data_path, save_path=train_data_path,
                                         pad_max_len=pad_max_len, token_dict=dict_path)
        logger.info("Begin preprocess valid data")
        tetrad_text_to_token_id_for_bert(file_path=raw_valid_data_path, save_path=valid_data_path,
                                         pad_max_len=pad_max_len, token_dict=dict_path)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = TetradDataGenerator(train_file.readlines(), batch_size)
            valid_generator = TetradDataGenerator(valid_file.readlines(), batch_size, random=False)

        bert_config = BertConfig.from_json_file(json_file_path=config_path)
        model = PolyEncoder(config=bert_config, batch_size=batch_size, bert_model_type=model_type,
                            poly_type=poly_type, candi_agg_type=candi_agg_type, poly_m=poly_m)

        weight_dict = load_bert_weights(
            model_file_path=model_file_path, model=model,
            mapping=bert_variable_mapping(bert_config.num_hidden_layers, prefix_="bert_model.")
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        pipeline = CustomPipeline([model], batch_size, device, torch.IntTensor, torch.LongTensor)
        history = {"t_acc": [], "t_loss": [], "v_acc": [], "v_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=seed)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)
            checkpoint = Checkpoint(checkpoint_dir=checkpoint_dir, optimizer=optimizer, model=model)

            pipeline.train(train_generator, valid_generator, epochs, optimizer,
                           checkpoint, checkpoint_save_freq, history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(model_dir="./data/ch/bert/chinese_wwm_pytorch", execute_type="train")
