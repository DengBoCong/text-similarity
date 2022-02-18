#! -*- coding: utf-8 -*-
""" TensorFlow Run SimCSE
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
from sim.tensorflow.pipeline import NormalPipeline
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


class SimCSEPipeline(NormalPipeline):
    def __init__(self, model: list, batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        """
        super(SimCSEPipeline, self).__init__(model, batch_size)

    def simcse_loss(self, pred: Any) -> tuple:
        """计算simcse lss
        """

        ids = keras.backend.arange(0, pred.shape[0])
        ids_1 = ids[None, :]
        ids_2 = (ids + 1 - ids % 2 * 2)[:, None]
        y_true = tf.equal(ids_1, ids_2)
        y_true = tf.cast(y_true, tf.float32)

        pred = tf.math.l2_normalize(pred, axis=1)
        sim = tf.matmul(pred, pred, transpose_b=True)
        sim = (sim - tf.eye(pred.shape[0]) * 1e12) * 20

        return y_true, sim

    def _train_step(self, batch_dataset: dict, optimizer: keras.optimizers.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        with tf.GradientTape() as tape:
            outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])
            y_true, sim = self.simcse_loss(outputs)
            loss = keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, sim)

        accuracy = keras.metrics.CategoricalAccuracy()(y_true, sim)

        variables = self.model[0].trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return {"train_loss": loss, "train_accuracy": accuracy}

    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 验证步
        :param batch_dataset: 验证步的当前batch数据
        """
        outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])
        y_true, sim = self.simcse_loss(outputs)
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, sim)
        accuracy = keras.metrics.CategoricalAccuracy()(y_true, sim)

        return {"train_loss": loss, "train_accuracy": accuracy}

    def inference(self, query1: str, query2: str) -> Any:
        """ 推断模块
        :param query1: 文本1
        :param query2: 文本2
        :return:
        """
        pass

    def _save_model(self, *args, **kwargs) -> NoReturn:
        pass


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
    checkpoint_path = os.path.join(model_dir, "bert_model.ckpt")
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
        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = SimCSEDataGenerator(train_file.readlines(), batch_size)
            valid_generator = SimCSEDataGenerator(valid_file.readlines(), batch_size, random=False)

        bert_config = BertConfig.from_json_file(json_file_path=config_path)

        with_pool = "linear" if pooling == "pooler" else False
        if model_type == "bert":
            bert = bert_model(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
            load_bert_weights_from_checkpoint(checkpoint_path, bert,
                                              bert_variable_mapping(bert_config.num_hidden_layers))
        elif model_type == "albert":
            bert = albert(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
            load_bert_weights_from_checkpoint(checkpoint_path, bert, albert_variable_mapping())
        elif model_type == "nezha":
            bert = NEZHA(config=bert_config, batch_size=batch_size * 2, with_pool=with_pool)
            load_bert_weights_from_checkpoint(checkpoint_path, bert,
                                              bert_variable_mapping(bert_config.num_hidden_layers))
        else:
            raise ValueError("model_type is None or not support")

        # 这里取一下第一层和最后一层
        if model_type == "albert":
            first_block_layer_output = bert.get_layer("bert-layer").output
            last_block_layer_output = bert.get_layer("bert-layer").output
        else:
            first_block_layer_output = bert.get_layer("bert-layer-0").output
            last_block_layer_output = bert.get_layer(f"bert-layer-{bert_config.num_hidden_layers - 1}").output

        if pooling == 'first-last-avg':
            outputs = keras.layers.Average()([
                keras.layers.GlobalAveragePooling1D()(first_block_layer_output),
                keras.layers.GlobalAveragePooling1D()(last_block_layer_output)
            ])
        elif pooling == "last-avg":
            outputs = keras.layers.GlobalAveragePooling1D()(last_block_layer_output)
        elif pooling == "cls":
            outputs = keras.layers.Lambda(lambda x: x[:, 0])(last_block_layer_output)
        elif pooling == "pooler":
            outputs = bert.output
        else:
            raise ValueError("pooling is None or not support")

        model = keras.Model(inputs=bert.inputs, outputs=outputs)
        checkpoint_manager = load_checkpoint(checkpoint_dir=checkpoint_dir, execute_type=execute_type,
                                             checkpoint_save_size=checkpoint_save_size, model=model)

        pipeline = SimCSEPipeline([model], batch_size)
        history = {"train_accuracy": [], "train_loss": [], "valid_accuracy": [], "valid_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=seed)
            optimizer = keras.optimizers.Adam(learning_rate=1e-5)

            pipeline.train(train_generator, valid_generator, epochs, optimizer,
                           checkpoint_manager, checkpoint_save_freq, history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        elif execute_type == "inference":
            pass
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(model_dir="./data/ch/bert/chinese_wwm_L-12_H-768_A-12", execute_type="train", model_type="bert")
