#! -*- coding: utf-8 -*-
""" word2vec
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl
import random

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
from typing import Any
from typing import NoReturn


class SimpleCallback(CallbackAny2Vec):
    """定义一下每个epoch打印loss"""

    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0.

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print("Loss after epoch {}: {}".format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


def get_tokenized_sentence(sentence: str, ngram_words: Any, max_ngram: int = 4) -> str:
    """根据ngram进行tokenized
    :param sentence: 文本
    :param ngram_words: ngram映射表
    :param max_ngram: max ngram
    :return:
    """
    output_tokens, sentence_len, count = [], len(sentence), 0
    while count < sentence_len:
        words = [sentence[count]]
        for ngram_num in range(2, max_ngram + 1):
            if sentence[count:count + ngram_num] in ngram_words:
                words.append(sentence[count:count + ngram_num])
        token = random.choice(words)
        output_tokens.append(token)
        count += len(token)

    return " ".join(output_tokens)


def train_word2vec_model(file_path: str,
                         save_path: str,
                         ngram_file: str,
                         vector_size: int = 128,
                         window: int = 5,
                         compute_loss: bool = True,
                         min_count: int = 1,
                         seed: int = 1,
                         workers: int = 32,
                         alpha: float = 0.5,
                         min_alpha: float = 0.0005,
                         epochs: int = 10,
                         batch_words: int = int(4e4),
                         callbacks: list = None) -> NoReturn:
    """根据ngram映射表将文本列表文件转化成token列表
    :param file_path: 文本文件路径
    :param save_path: 模型保存路径
    :param ngram_file: ngram文件路径
    :param vector_size: 维度
    :param window: 移动窗体大小
    :param compute_loss: 是否保存计算损失
    :param min_count: 最小词频数
    :param seed: 随机种子
    :param workers: 训练线程数
    :param alpha: 初始化学习率
    :param min_alpha: 最小学习率
    :param epochs: 训练轮数
    :param batch_words: 一次传给工作线程的训练数量
    :param callbacks: 训练回调对象，[SimpleCallback()]
    :return:
    """
    if callbacks is None:
        callbacks = [SimpleCallback()]
    ngram_words = pkl.load(open(ngram_file, "rb"))
    sentences = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            line = line.strip().strip("\n")
            cur_sentences = []
            for i in range(10):
                output_sentence = get_tokenized_sentence(line, ngram_words)
                if output_sentence not in cur_sentences:
                    cur_sentences.append(output_sentence)

            sentences.extend(cur_sentences)

        sentences = [sentence.split() for sentence in sentences]
        random.shuffle(sentences)

        model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, compute_loss=compute_loss,
                         min_count=min_count, seed=seed, workers=workers, alpha=alpha, min_alpha=min_alpha,
                         epochs=epochs, batch_words=batch_words, callbacks=callbacks)

        model.wv.save(save_path)
