#! -*- coding: utf-8 -*-
""" TensorFlow Run Siamese RNN
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf
import tensorflow.keras as keras
from datetime import datetime
from sim.tensorflow.modeling_siamese_rnn import siamese_rnn_with_embedding
from sim.tools.data_processor.data_format import NormalDataGenerator
from sim.tools.data_processor.process_plain_text import text_pair_to_token_id
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tensorflow.common import load_checkpoint
from sim.tensorflow.common import set_seed
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from sim.tools.pipeline import NormalPipeline
from typing import Any
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


class TextPairPipeline(NormalPipeline):
    def __init__(self,
                 model: list,
                 batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param batch_size: batch size
        """
        super(TextPairPipeline, self).__init__(model, batch_size)

    def _train_step(self, batch_dataset: dict, optimizer: keras.optimizers.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        with tf.GradientTape() as tape:
            outputs1, outputs2 = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])

            diff = tf.reduce_sum(tf.abs(tf.math.subtract(outputs1, outputs2)), axis=1)
            sim = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
            pred = tf.square(tf.math.subtract(sim, batch_dataset["labels"]))
            loss = tf.reduce_sum(pred)

        accuracy = keras.metrics.BinaryAccuracy()(batch_dataset["labels"], sim)

        variables = self.model[0].trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return {"train_loss": loss / self.batch_size, "train_accuracy": accuracy}

    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 验证步
        :param batch_dataset: 验证步的当前batch数据
        """
        outputs1, outputs2 = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])

        diff = tf.reduce_sum(tf.abs(tf.math.subtract(outputs1, outputs2)), axis=1)
        sim = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
        pred = tf.square(tf.math.subtract(sim, batch_dataset["labels"]))
        loss = tf.reduce_sum(pred)
        accuracy = keras.metrics.BinaryAccuracy()(batch_dataset["labels"], sim)

        return {"train_loss": loss / self.batch_size, "train_accuracy": accuracy}

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
    key = str(datetime.now())
    logger.info("========================{}========================".format(key))
    # 训练时保存模型配置
    if execute_type == "train" and not save_model_config(key=key, model_desc="RNN Base",
                                                         model_config=options, config_path=MODEL_CONFIG_FILE_PATH):
        raise EOFError("An error occurred while saving the configuration file")

    if execute_type == "preprocess":
        logger.info("Begin preprocess train data")
        tokenizer = text_pair_to_token_id(file_path=options["raw_train_data_path"],
                                          save_path=options["train_data_path"], pad_max_len=options["vec_dim"])
        logger.info("Begin preprocess valid data")
        text_pair_to_token_id(file_path=options["raw_valid_data_path"],
                              save_path=options["valid_data_path"], pad_max_len=options["vec_dim"], tokenizer=tokenizer)
    else:
        with open(options["train_data_path"], "r", encoding="utf-8") as train_file, open(
                options["valid_data_path"], "r", encoding="utf-8") as valid_file:
            train_generator = NormalDataGenerator(train_file.readlines(), options["batch_size"])
            valid_generator = NormalDataGenerator(valid_file.readlines(), options["batch_size"], random=False)

        model = siamese_rnn_with_embedding(emb_dim=options["embedding_dim"], vec_dim=options["vec_dim"],
                                           vocab_size=options["vocab_size"], units=options["units"],
                                           cell_type=options["rnn"], share=options["share"])
        checkpoint_manager = load_checkpoint(checkpoint_dir=options["checkpoint_dir"], execute_type=execute_type,
                                             checkpoint_save_size=options["checkpoint_save_size"], model=model)

        pipeline = TextPairPipeline([model], options["batch_size"])
        history = {"train_accuracy": [], "train_loss": [], "valid_accuracy": [], "valid_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=options["seed"])
            optimizer = keras.optimizers.Adam(name="optimizer")

            pipeline.train(train_generator, valid_generator, options["epochs"], optimizer,
                           checkpoint_manager, options["checkpoint_save_freq"], history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        elif execute_type == "inference":
            pass
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(config_path="./data/config/siamse_rnn.json", execute_type="preprocess")
