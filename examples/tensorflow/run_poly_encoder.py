#! -*- coding: utf-8 -*-
""" TensorFlow Run Poly Encoder
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow as tf
import tensorflow.keras as keras
from datetime import datetime
from sim.tensorflow import albert_variable_mapping
from sim.tensorflow import bert_variable_mapping
from sim.tensorflow.common import load_bert_weights_from_checkpoint
from sim.tensorflow.common import load_checkpoint
from sim.tensorflow.common import set_seed
from sim.tensorflow.modeling_albert import albert
from sim.tensorflow.modeling_bert import bert_model
from sim.tensorflow.modeling_nezha import NEZHA
from sim.tensorflow.modeling_poly_encoder import poly_encoder
from sim.tensorflow.pipeline import TextPairPipeline
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
tf.config.run_functions_eagerly(True)


class CustomPipeline(TextPairPipeline):
    def __init__(self, model: list, batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        """
        super(CustomPipeline, self).__init__(model, batch_size)

    @tf.function(autograph=True)
    def _train_step(self, batch_dataset: dict, optimizer: keras.optimizers.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        with tf.GradientTape() as tape:
            outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"],
                                            batch_dataset["inputs3"], batch_dataset["inputs4"]])
            loss, accuracy = self._metrics(batch_dataset["labels"], outputs)

        variables = self.model[0].trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return {"t_loss": loss, "t_acc": accuracy}

    @tf.function(autograph=True)
    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 验证步
        :param batch_dataset: 验证步的当前batch数据
        """
        outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"],
                                        batch_dataset["inputs3"], batch_dataset["inputs4"]])
        loss, accuracy = self._metrics(batch_dataset["labels"], outputs)

        return {"v_loss": loss, "v_acc": accuracy}


def actuator(execute_type: str, model_type: str, model_dir: str = None) -> NoReturn:
    """
    :param execute_type: 执行类型，"bert", "albert", "nezha", "word2vec", "emb"
    :param model_type: 模型类型
    :param model_dir: 预训练模型目录，如果不用预训练模型不传
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
    checkpoint_path = os.path.join(model_dir, "bert_model.ckpt")
    dict_path = os.path.join(model_dir, "vocab.txt")

    with open(config_path, "r", encoding="utf-8") as file:
        options = json.load(file)

    # 这里在日志文件里面做一个执行分割
    key = str(datetime.now())
    logger.info("========================{}========================".format(key))
    # 训练时保存模型配置
    if execute_type == "train" and not save_model_config(key=key, model_desc="Poly Encoder",
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
        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = TetradDataGenerator(train_file.readlines(), batch_size)
            valid_generator = TetradDataGenerator(valid_file.readlines(), batch_size, random=False)

        # 这里使用bert作为Embedding
        if model_type == "bert":
            bert_config = BertConfig.from_json_file(json_file_path=config_path)
            bert = bert_model(config=bert_config, batch_size=batch_size)
            load_bert_weights_from_checkpoint(checkpoint_path, bert,
                                              bert_variable_mapping(bert_config.num_hidden_layers))
        elif model_type == "albert":
            bert_config = BertConfig.from_json_file(json_file_path=config_path)
            bert = albert(config=bert_config, batch_size=batch_size)
            load_bert_weights_from_checkpoint(checkpoint_path, bert, albert_variable_mapping())
        elif model_type == "nezha":
            bert_config = BertConfig.from_json_file(json_file_path=config_path)
            bert = NEZHA(config=bert_config, batch_size=batch_size)
            load_bert_weights_from_checkpoint(checkpoint_path, bert,
                                              bert_variable_mapping(bert_config.num_hidden_layers))
        else:
            raise ValueError("`model_type` must in bert/albert/nezha")
        # 可开可不开
        # bert.trainable = False

        model = poly_encoder(context_bert_model=bert, candidate_bert_model=bert, batch_size=batch_size,
                             embeddings_size=bert_config.hidden_size, poly_type=poly_type,
                             candi_agg_type=candi_agg_type, poly_m=poly_m)

        checkpoint_manager = load_checkpoint(checkpoint_dir=checkpoint_dir, execute_type=execute_type,
                                             checkpoint_save_size=checkpoint_save_size, model=model)

        pipeline = CustomPipeline([model], batch_size)
        history = {"t_acc": [], "t_loss": [], "v_acc": [], "v_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=seed)
            optimizer = keras.optimizers.Adam(learning_rate=2e-5)

            pipeline.train(train_generator, valid_generator, epochs, optimizer,
                           checkpoint_manager, checkpoint_save_freq, history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(execute_type="trian", model_type="bert", model_dir="./data/ch/bert/chinese_wwm_L-12_H-768_A-12")
