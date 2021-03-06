#! -*- coding: utf-8 -*-
""" TensorFlow Run CNN Base
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
from sim.tensorflow.modeling_text_cnn import text_cnn
from sim.tensorflow.modeling_nezha import NEZHA
from sim.tensorflow.pipeline import TextPairPipeline
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import NormalDataGenerator
from sim.tools.data_processor.process_ngram import construct_ngram_dict
from sim.tools.data_processor.process_plain_text import text_pair_to_token_id
from sim.tools.data_processor.process_plain_text import text_to_token_id_for_bert
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from sim.tools.word2vec import train_word2vec_model
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)
tf.config.run_functions_eagerly(True)


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
    raw_train_data_path = "./corpus/chinese/LCQMC/train.txt"
    raw_valid_data_path = "./corpus/chinese/LCQMC/test.txt"
    train_data_path = "./data/train1.txt"
    valid_data_path = "./data/test1.txt"
    checkpoint_dir = "./data/checkpoint/"
    checkpoint_save_size = 5
    checkpoint_save_freq = 2

    # Text CNN模型配置
    units = 2
    filter_num = 300
    kernel_sizes = [3, 4, 5]
    initializers = ["normal", "normal", "normal"]
    activations = ["tanh", "tanh", "tanh"]
    padding = "valid"

    # 如果使用预训练模型，这里自行修改一下啦
    config_path = os.path.join(model_dir, "bert_config.json")
    checkpoint_path = os.path.join(model_dir, "bert_model.ckpt")
    dict_path = os.path.join(model_dir, "vocab.txt")

    # 如果用的话Word2Vec，这里就是模型保存路径
    text_file_path = "./data/all_texts.txt"
    ngram_dict_file_path = "./data/ngram.pkl"
    word2vec_file_path = "./data/word2vec.model"

    # 如果是使用Tokenizer，这里就保存Tokenizer的路径就可以了
    tokenizer_file_path = "./data/tokenizer.json"

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
        if model_type in ["bert", "albert", "nezha"]:
            logger.info("Begin preprocess train data")
            text_to_token_id_for_bert(file_path=raw_train_data_path, save_path=train_data_path,
                                      pad_max_len=pad_max_len, token_dict=dict_path)
            logger.info("Begin preprocess valid data")
            text_to_token_id_for_bert(file_path=raw_valid_data_path, save_path=valid_data_path,
                                      pad_max_len=pad_max_len, token_dict=dict_path)
        elif model_type == "word2vec":
            # 可以先训一个w2v模型
            construct_ngram_dict(text_file_path, ngram_dict_file_path)
            train_word2vec_model(text_file_path, word2vec_file_path, ngram_dict_file_path)
        else:
            logger.info("Begin preprocess train data")
            tokenizer = text_pair_to_token_id(file_path=raw_train_data_path,
                                              save_path=train_data_path, pad_max_len=pad_max_len)
            with open(tokenizer_file_path, "w", encoding="utf-8") as file:
                file.write(tokenizer.to_json())
            logger.info("Begin preprocess valid data")
            text_pair_to_token_id(file_path=raw_valid_data_path,
                                  save_path=valid_data_path, pad_max_len=pad_max_len, tokenizer=tokenizer)
    else:
        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = NormalDataGenerator(train_file.readlines(), batch_size)
            valid_generator = NormalDataGenerator(valid_file.readlines(), batch_size, random=False)

        # 这里使用bert作为Embedding
        if model_type == "bert":
            bert_config = BertConfig.from_json_file(json_file_path=config_path)
            bert = bert_model(config=bert_config, batch_size=batch_size)
            load_bert_weights_from_checkpoint(checkpoint_path, bert, bert_variable_mapping(bert_config.num_hidden_layers))
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
        bert.trainable = False

        outputs = text_cnn(pad_max_len, bert_config.hidden_size, units, filter_num,
                           kernel_sizes, initializers, activations, padding)(bert.outputs)
        model = keras.Model(inputs=bert.inputs, outputs=outputs)

        checkpoint_manager = load_checkpoint(checkpoint_dir=checkpoint_dir, execute_type=execute_type,
                                             checkpoint_save_size=checkpoint_save_size, model=model)

        pipeline = TextPairPipeline([model], batch_size)
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
