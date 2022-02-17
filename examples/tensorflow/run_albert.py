#! -*- coding: utf-8 -*-
""" TensorFlow Run Albert
"""
# Author: DengBoCong <bocongdeng@gmail.com>
# 中文与训练模型：https://github.com/brightmart/albert_zh
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow.keras as keras
from datetime import datetime
from sim.tensorflow import albert_variable_mapping
from sim.tensorflow.common import load_bert_weights_from_checkpoint
from sim.tensorflow.common import load_checkpoint
from sim.tensorflow.common import set_seed
from sim.tensorflow.modeling_albert import albert
from sim.tensorflow.optimizers import PiecewiseLinearDecay
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import NormalDataGenerator
from sim.tools.data_processor.process_plain_text import text_to_token_id_for_bert
from sim.tensorflow.pipeline import TextPairPipeline
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


def actuator(model_dir: str, execute_type: str) -> NoReturn:
    """
    :param model_dir: 预训练模型目录
    :param execute_type: 执行类型
    """
    batch_size = 64
    pad_max_len = 40
    seed = 1
    epochs = 5
    checkpoint_save_size = 5
    checkpoint_save_freq = 2
    raw_train_data_path = "./corpus/chinese/LCQMC/train.txt"
    raw_valid_data_path = "./corpus/chinese/LCQMC/test.txt"
    train_data_path = "./data/train1.txt"
    valid_data_path = "./data/test1.txt"
    checkpoint_dir = "./data/checkpoint/"

    config_path = os.path.join(model_dir, "albert_config_small_google.json")
    checkpoint_path = os.path.join(model_dir, "albert_model.ckpt")
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
        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = NormalDataGenerator(train_file.readlines(), batch_size)
            valid_generator = NormalDataGenerator(valid_file.readlines(), batch_size, random=False)

        bert_config = BertConfig.from_json_file(json_file_path=config_path)
        bert = albert(config=bert_config, batch_size=batch_size)

        load_bert_weights_from_checkpoint(checkpoint_path, bert, albert_variable_mapping())

        outputs = keras.layers.Lambda(lambda x: x[:, 0], name="cls-token")(bert.output)
        outputs = keras.layers.Dense(
            units=2, activation="softmax", kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)
        )(outputs)
        model = keras.Model(inputs=bert.input, outputs=outputs)

        checkpoint_manager = load_checkpoint(checkpoint_dir=checkpoint_dir, execute_type=execute_type,
                                             checkpoint_save_size=checkpoint_save_size, model=model)

        pipeline = TextPairPipeline([model], batch_size)
        history = {"train_accuracy": [], "train_loss": [], "valid_accuracy": [], "valid_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=seed)
            lr_scheduler = PiecewiseLinearDecay(boundaries=[1000, 2000], values=[1., 0.1])
            optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)

            pipeline.train(train_generator, valid_generator, epochs, optimizer,
                           checkpoint_manager, checkpoint_save_freq, history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        elif execute_type == "inference":
            pass
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(model_dir="./data/ch/bert/albert_small_zh_google", execute_type="train")
