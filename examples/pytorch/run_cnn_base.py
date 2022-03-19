#! -*- coding: utf-8 -*-
""" Pytorch Run Fast Text
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
from sim.pytorch.common import truncated_normal_
from sim.pytorch.modeling_albert import ALBERT
from sim.pytorch.modeling_bert import BertModel
from sim.pytorch.modeling_text_cnn import TextCNN
from sim.pytorch.modeling_nezha import NEZHA
from sim.pytorch.pipeline import TextPairPipeline
from sim.tools import BertConfig
from sim.tools.data_processor.data_format import NormalDataGenerator
from sim.tools.data_processor.process_ngram import construct_ngram_dict
from sim.tools.data_processor.process_plain_text import text_pair_to_token_id
from sim.tools.data_processor.process_plain_text import text_to_token_id_for_bert
from sim.tools.data_processor.process_plain_text import text_to_token_id_for_bert
from sim.tools.settings import MODEL_CONFIG_FILE_PATH
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sim.tools.tools import save_model_config
from sim.tools.word2vec import train_word2vec_model
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


class Model(nn.Module):
    """组合模型适配任务
    """

    def __init__(self,
                 bert_config: BertConfig,
                 batch_size: int,
                 seq_len: int,
                 filter_num: int,
                 kernel_sizes: list = None,
                 activations: list = None,
                 model_type: str = "bert"):
        super(Model, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        if activations is None:
            activations = ["tanh", "tanh", "tanh"]

        if model_type == "bert":
            self.bert_model = BertModel(config=bert_config, batch_size=batch_size)
        elif model_type == "albert":
            self.bert_model = ALBERT(config=bert_config, batch_size=batch_size)
        elif model_type == "nezha":
            self.bert_model = NEZHA(config=bert_config, batch_size=batch_size)
        else:
            raise ValueError("`model_type` must in bert/albert/nezha")

        for params in self.bert_model.parameters():
            params.requires_grad = False  # 固定权重

        self.text_cnn = TextCNN(seq_len, bert_config.hidden_size, units=2, filter_num=filter_num,
                                kernel_sizes=kernel_sizes, activations=activations)

    def forward(self, input_ids, token_type_ids):
        outputs = self.bert_model(input_ids, token_type_ids)
        outputs = self.text_cnn(outputs)

        return outputs


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
    raw_train_data_path = "./corpus/chinese/LCQMC/train.txt"
    raw_valid_data_path = "./corpus/chinese/LCQMC/test.txt"
    train_data_path = "./data/train1.txt"
    valid_data_path = "./data/test1.txt"
    checkpoint_dir = "./data/checkpoint/"
    checkpoint_save_size = 5
    checkpoint_save_freq = 2

    # 如果使用预训练模型，这里自行修改一下啦
    config_path = os.path.join(model_dir, "bert_config.json")
    model_file_path = os.path.join(model_dir, "pytorch_model.bin")
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(train_data_path, "r", encoding="utf-8") as train_file, open(
                valid_data_path, "r", encoding="utf-8") as valid_file:
            train_generator = NormalDataGenerator(train_file.readlines(), batch_size)
            valid_generator = NormalDataGenerator(valid_file.readlines(), batch_size, random=False)

        bert_config = BertConfig.from_json_file(json_file_path=config_path)
        model = Model(bert_config=bert_config, batch_size=batch_size, seq_len=pad_max_len, filter_num=300)

        weight_dict = load_bert_weights(
            model_file_path=model_file_path, model=model,
            mapping=bert_variable_mapping(bert_config.num_hidden_layers, prefix_="bert_model.")
        )
        model.load_state_dict(state_dict=weight_dict)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2])
            model.to(device)

        pipeline = TextPairPipeline([model], batch_size, device, torch.IntTensor, torch.LongTensor)
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
