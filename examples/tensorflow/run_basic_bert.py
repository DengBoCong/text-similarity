#! -*- coding: utf-8 -*-
""" TensorFlow Run Basic Bert
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
from sim.tensorflow.common import load_bert_weights_from_checkpoint
from sim.tensorflow.common import load_checkpoint
from sim.tensorflow.common import set_seed
from sim.tensorflow.modeling_bert import bert_model
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import NormalDataGenerator
from sim.tools.data_processor.process_plain_text import text_to_token_id_for_bert
from sim.tools.pipeline import NormalPipeline
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from typing import Any
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


def variable_mapping(num_hidden_layers):
    """映射到官方BERT权重格式
    :param num_hidden_layers: encoder的层数
    """
    mapping = {
        "embedding-token/embeddings": "bert/embeddings/word_embeddings",
        "embedding-segment/embeddings": "bert/embeddings/token_type_embeddings",
        "embedding-position/embeddings": "bert/embeddings/position_embeddings",
        "embedding-norm/gamma": "bert/embeddings/LayerNorm/gamma",
        "embedding-norm/beta": "bert/embeddings/LayerNorm/beta",
        "embedding-mapping/kernel": "bert/encoder/embedding_hidden_mapping_in/kernel",
        "embedding-mapping/bias": "bert/encoder/embedding_hidden_mapping_in/bias",
        "bert-output/pooler-dense/kernel": "bert/pooler/dense/kernel",
        "bert-output/pooler-dense/bias": "bert/pooler/dense/bias",
        "bert-output/nsp-prob/kernel": "cls/seq_relationship/output_weights",
        "bert-output/nsp-prob/bias": "cls/seq_relationship/output_bias",
        "bert-output/mlm-dense/kernel": "cls/predictions/transform/dense/kernel",
        "bert-output/mlm-dense/bias": "cls/predictions/transform/dense/bias",
        "bert-output/mlm-norm/gamma": "cls/predictions/transform/LayerNorm/gamma",
        "bert-output/mlm-norm/beta": "cls/predictions/transform/LayerNorm/beta",
        "bert-output/mlm-bias/bias": "cls/predictions/output_bias"
    }

    for i in range(num_hidden_layers):
        prefix = 'bert/encoder/layer_%d/' % i
        mapping.update({
            f"bert-layer-{i}/multi-head-self-attention/query/kernel": prefix + "attention/self/query/kernel",
            f"bert-layer-{i}/multi-head-self-attention/query/bias": prefix + "attention/self/query/bias",
            f"bert-layer-{i}/multi-head-self-attention/key/kernel": prefix + "attention/self/key/kernel",
            f"bert-layer-{i}/multi-head-self-attention/key/bias": prefix + "attention/self/key/bias",
            f"bert-layer-{i}/multi-head-self-attention/value/kernel": prefix + "attention/self/value/kernel",
            f"bert-layer-{i}/multi-head-self-attention/value/bias": prefix + "attention/self/value/bias",
            f"bert-layer-{i}/multi-head-self-attention/output/kernel": prefix + "attention/output/dense/kernel",
            f"bert-layer-{i}/multi-head-self-attention/output/bias": prefix + "attention/output/dense/bias",
            f"bert-layer-{i}/multi-head-self-attention-norm/gamma": prefix + "attention/output/LayerNorm/gamma",
            f"bert-layer-{i}/multi-head-self-attention-norm/beta": prefix + "attention/output/LayerNorm/beta",
            f"bert-layer-0/feedforward/input/kernel": prefix + "intermediate/dense/kernel",
            f"bert-layer-0/feedforward/input/bias": prefix + "intermediate/dense/bias",
            f"bert-layer-0/feedforward/output/kernel": prefix + "output/dense/kernel",
            f"bert-layer-0/feedforward/output/bias": prefix + "output/dense/bias",
            f"bert-layer-0/feedforward-norm/gamma": prefix + "output/LayerNorm/gamma",
            f"bert-layer-0/feedforward-norm/beta": prefix + "output/LayerNorm/beta",
        })

    return mapping


class TextPairPipeline(NormalPipeline):
    def __init__(self,
                 model: list,
                 loss_metric: keras.metrics.Metric,
                 accuracy_metric: keras.metrics.Metric,
                 batch_size: int):
        """
        :param model: 模型相关组件，用于train_step和valid_step中自定义使用
        :param loss_metric: 损失计算器，必传指标
        :param accuracy_metric: 精度计算器，必传指标
        :param batch_size: batch size
        """
        super(TextPairPipeline, self).__init__(model, loss_metric, accuracy_metric, batch_size)

    def _train_step(self, batch_dataset: dict, optimizer: keras.optimizers.Optimizer, *args, **kwargs) -> dict:
        """ 训练步
        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        with tf.GradientTape() as tape:
            outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])
            loss = keras.losses.SparseCategoricalCrossentropy()(batch_dataset["labels"], outputs)

        accuracy = keras.metrics.SparseCategoricalAccuracy()([[label] for label in batch_dataset["labels"]], outputs)
        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(accuracy)

        variables = self.model[0].trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return {"train_loss": self.loss_metric.result(), "train_accuracy": self.accuracy_metric.result()}

    def _valid_step(self, batch_dataset: dict, *args, **kwargs) -> dict:
        """ 验证步
        :param batch_dataset: 验证步的当前batch数据
        """
        outputs = self.model[0](inputs=[batch_dataset["inputs1"], batch_dataset["inputs2"]])
        loss = keras.losses.SparseCategoricalCrossentropy()(batch_dataset["labels"], outputs)
        accuracy = keras.metrics.SparseCategoricalAccuracy()([[label] for label in batch_dataset["labels"]], outputs)
        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(accuracy)

        return {"train_loss": self.loss_metric.result(), "train_accuracy": self.accuracy_metric.result()}

    def inference(self, query1: str, query2: str) -> Any:
        """ 推断模块
        :param query1: 文本1
        :param query2: 文本2
        :return:
        """
        pass

    def _save_model(self, *args, **kwargs) -> NoReturn:
        pass


def actuator(model_dir: str, execute_type: str) -> NoReturn:
    """
    :param model_dir: 预训练模型目录
    :param execute_type: 执行类型
    """
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
        text_to_token_id_for_bert(file_path=options["raw_train_data_path"], save_path=options["train_data_path"],
                                  pad_max_len=options["pad_max_len"], token_dict=dict_path)
        logger.info("Begin preprocess valid data")
        text_to_token_id_for_bert(file_path=options["raw_valid_data_path"], save_path=options["valid_data_path"],
                                  pad_max_len=options["pad_max_len"], token_dict=dict_path)
    else:
        with open(options["train_data_path"], "r", encoding="utf-8") as train_file, open(
                options["valid_data_path"], "r", encoding="utf-8") as valid_file:
            train_generator = NormalDataGenerator(train_file.readlines(), options["batch_size"])
            valid_generator = NormalDataGenerator(valid_file.readlines(), options["batch_size"])

        bert_config = BertConfig.from_json_file(json_file_path=config_path)
        bert = bert_model(config=bert_config, batch_size=options["batch_size"])
        load_bert_weights_from_checkpoint(checkpoint_path, bert, variable_mapping(bert_config.num_hidden_layers))

        outputs = keras.layers.Dropout(rate=0.1)(bert.output)
        outputs = keras.layers.Dense(
            units=2, activation="softmax", kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)
        )(outputs)
        model = keras.Model(inputs=bert.input, outputs=outputs)

        checkpoint_manager = load_checkpoint(checkpoint_dir=options["checkpoint_dir"], execute_type=execute_type,
                                             checkpoint_save_size=options["checkpoint_save_size"], model=model)

        loss_metric = keras.metrics.Mean()
        accuracy_metric = keras.metrics.Mean()
        pipeline = TextPairPipeline([model], loss_metric, accuracy_metric, options["batch_size"])
        history = {"train_accuracy": [], "train_loss": [], "valid_accuracy": [], "valid_loss": []}

        if execute_type == "train":
            set_seed(manual_seed=options["seed"])
            optimizer = keras.optimizers.Adam(learning_rate=2e-5)

            pipeline.train(train_generator, valid_generator, options["epochs"], optimizer,
                           checkpoint_manager, options["checkpoint_save_freq"], history)
        elif execute_type == "evaluate":
            pipeline.evaluate(valid_generator, history)
        elif execute_type == "inference":
            pass
        else:
            raise ValueError("execute_type error")


if __name__ == '__main__':
    actuator(model_dir="./data/ch/bert/chinese_wwm_L-12_H-768_A-12", execute_type="train")
