#! -*- coding: utf-8 -*-
""" RNN Base Entrance
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RNN_BASE_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RNN_BASE_LOG_FILE_PATH, formatter="%(message)s")


def actuator() -> NoReturn:
    parser = ArgumentParser(description="执行器")
    parser.add_argument("--type", default="tf", type=str, required=False, help="计算框架类型，tf/torch")
    parser.add_argument("--execute_type", default="preprocess", type=str, required=False, help="执行模式")
    parser.add_argument("--train_desc", default="", type=str, required=False, help="训练备注")
    parser.add_argument("--embedding_dim", default=512, type=int, required=False, help="词嵌入大小")
    parser.add_argument("--seed", default=1, type=int, required=False, help="随机种子")
    parser.add_argument("--vec_dim", default=12, type=int, required=False, help="最大句子序列长度")
    parser.add_argument("--vocab_size", default=40000, type=int, required=False, help="词汇量大小")
    parser.add_argument("--units", default=1024, type=int, required=False, help="RNN输出单元大小")
    parser.add_argument("--rnn", default="lstm", type=str, required=False, help="RNN实现类型")
    parser.add_argument("--share", default=False, type=bool, required=False, help="是否共享参数")
    parser.add_argument("--checkpoint_dir", default="./data/checkpoint/", type=str, required=False, help="检查点保存相对路径")
    parser.add_argument("--checkpoint_save_size", default=5, type=int, required=False, help="最大保存检查点数量")
    parser.add_argument("--checkpoint_save_freq", default=2, type=int, required=False, help="检查点保存频率")
    parser.add_argument("--raw_train_data_path", default="./corpus/chinese/LCQMC/train.txt",
                        type=str, required=False, help="原始训练数据路径")
    parser.add_argument("--raw_valid_data_path", default="./corpus/chinese/LCQMC/test.txt",
                        type=str, required=False, help="原始验证数据路径")
    parser.add_argument("--train_data_path", default="./data/train.txt", type=str, required=False, help="处理后的训练数据路径")
    parser.add_argument("--valid_data_path", default="./data/test.txt", type=str, required=False, help="处理后的验证数据路径")
    parser.add_argument("--batch_size", default=64, type=int, required=False, help="batch大小")
    parser.add_argument("--epochs", default=5, type=int, required=False, help="训练步数")
    parser.add_argument("--num_layers", default=2, type=int, required=False, help="层数")
    parser.add_argument("--bi", default=True, type=bool, required=False, help="是否双层")
    parser.add_argument("--dropout", default=0.2, type=float, required=False, help="采样率")

    options = parser.parse_args()
    actuator_ = import_module('sim.rnn_base.{}.actuator'.format(options.type))

    # 这里在日志文件里面做一个执行分割
    key = int(datetime.now().timestamp())
    logger.info("\n========================{}========================\n".format(key))
    # 训练时保存模型配置
    if options.execute_type == "train" and not save_model_config(key=str(key), model_desc="RNN Base",
                                                                 options=options, config_path=MODEL_CONFIG_FILE_PATH):
        raise EOFError("An error occurred while saving the configuration file")

    actuator_.actuator(options)
