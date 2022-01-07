#! -*- coding: utf-8 -*-
""" Pytorch Run Siamese RNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os
import random
import torch.optim

from datetime import datetime
from sim.pytorch.siamese_rnn import SiameseRnnWithEmbedding
from sim.tools.data_processor.process_plain_text import datasets_generator
from sim.tools.data_processor.process_plain_text import text_pair_to_token_id
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from sim.pytorch.common import Checkpoint
from sim.tools.pipeline import NormalPipeline
from typing import Any
from typing import NoReturn


logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


class TextPairPipeline(NormalPipeline):
    def __init__(self, model: list, loss_metric: Any, accuracy_metric: Any, batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param loss_metric: 损失计算器，必传指标
        :param accuracy_metric: 精度计算器，必传指标
        :param batch_size: batch size
        """
        super(TextPairPipeline, self).__init__(model, loss_metric, accuracy_metric, batch_size)

    def _train_step(self, batch_dataset: tuple, optimizer: torch.optim.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        inputs1 = torch.from_numpy(batch_dataset[0]).permute(1, 0)
        inputs2 = torch.from_numpy(batch_dataset[1]).permute(1, 0)
        labels = torch.from_numpy(batch_dataset[2])

        optimizer.zero_grad()
        state1, state2 = self.model[0](inputs1, inputs2)

        diff = torch.sum(torch.abs(torch.sub(state1, state2)), dim=1)
        sim = torch.exp(-1.0 * diff)
        pred = torch.square(torch.sub(sim, labels))
        loss = torch.sum(pred)

        loss.backward()
        optimizer.step()

        return {"train_loss": torch.div(loss, self.batch_size), "train_accuracy": 0}

    def _valid_step(self, dataset: tuple, *args, **kwargs) -> dict:
        """ 验证步
        :param dataset: 训练步的当前batch数据
        """
        with torch.no_grad():
            inputs1 = torch.from_numpy(dataset[0]).permute(1, 0)
            inputs2 = torch.from_numpy(dataset[1]).permute(1, 0)
            labels = torch.from_numpy(dataset[2])

            state1, state2 = self.model[0](inputs1, inputs2)

            diff = torch.sum(torch.abs(torch.sub(state1, state2)), dim=1)
            sim = torch.exp(-1.0 * diff)
            pred = torch.square(torch.sub(sim, labels))
            loss = torch.sum(pred)

        return {"train_loss": torch.div(loss, self.batch_size), "train_accuracy": 0}

    def inference(self, query1: str, query2: str) -> Any:
        """ 推断模块
        :param query1: 文本1
        :param query2: 文本2
        :return:
        """
        pass

    def _save_model(self, *args, **kwargs) -> NoReturn:
        pass


def actuator(config_path: str, execute_type: str) -> NoReturn:
    """
    :param config_path: 配置json文件路径
    :param execute_type: 执行类型
    """
    with open(config_path, "r", encoding="utf-8") as file:
        options = json.load(file)

    # 这里在日志文件里面做一个执行分割
    key = int(datetime.now().timestamp())
    logger.info("========================{}========================".format(key))
    # 训练时保存模型配置
    if execute_type == "train" and not save_model_config(key=str(key), model_desc="RNN Base",
                                                         model_config=options, config_path=MODEL_CONFIG_FILE_PATH):
        raise EOFError("An error occurred while saving the configuration file")

    if execute_type == "preprocess":
        logger.info("Begin preprocess train data")
        tokenizer = text_pair_to_token_id(file_path=options["raw_train_data_path"],
                                          save_path=options["train_data_path"], pad_max_len=options["vec_dim"])
        logger.info("Begin preprocess valid data")
        text_pair_to_token_id(file_path=options["raw_valid_data_path"], save_path=options["valid_data_path"],
                              pad_max_len=options["vec_dim"], tokenizer=tokenizer)
    else:
        model = SiameseRnnWithEmbedding(emb_dim=options["embedding_dim"], vocab_size=options["vocab_size"],
                                        units=options["units"], dropout=options["dropout"],
                                        num_layers=options["num_layers"], rnn=options["rnn"],
                                        share=options["share"], if_bi=options["bi"])

        pipeline = TextPairPipeline([model], None, None, options["batch_size"])
        history = {"train_accuracy": [], "train_loss": [], "valid_accuracy": [], "valid_loss": []}

        if execute_type == "train":
            random.seed(options["seed"])
            os.environ['PYTHONHASHSEED'] = str(options["seed"])
            np.random.seed(options["seed"])
            torch.manual_seed(options["seed"])
            torch.cuda.manual_seed(options["seed"])
            torch.cuda.manual_seed_all(options["seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

            optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 1e-3}])
            checkpoint = Checkpoint(checkpoint_dir=options["checkpoint_dir"], optimizer=optimizer, model=model)
            pipeline.train(options["train_data_path"], options["valid_data_path"], options["epochs"],
                           optimizer, checkpoint, options["checkpoint_save_freq"], datasets_generator, history)
        elif execute_type == "evaluate":
            pipeline.evaluate(options["valid_data_path"], datasets_generator, history)
        elif execute_type == "inference":
            pass
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(config_path="./data/config/siamse_rnn.json", execute_type="preprocess")
