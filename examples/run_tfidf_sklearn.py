#! -*- coding: utf-8 -*-
""" Run TFIdf with sklearn
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from datetime import datetime
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Any
from typing import NoReturn

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


def infer(samples: list,
          labels: list,
          infer_size: int,
          ngram_range: tuple = (1, 5),
          k_fold: int = 5,
          manual_seed: int = 1) -> np.ndarray:
    """
    :param samples: 样本list，包含推断数据
    :param labels: 训练数据集标签
    :param infer_size: 推断数据集大小
    :param ngram_range: 词袋范围
    :param k_fold: k折
    :param manual_seed: 随机种子
    """
    tf_idf = TfidfVectorizer(ngram_range=ngram_range)
    tf_idf_feature = tf_idf.fit_transform(samples).toarray()
    # svd_feature = TruncatedSVD(n_components=100).fit_transform(tf_idf_feature)
    train_data = tf_idf_feature[:-infer_size]
    infer_data = tf_idf_feature[-infer_size:]

    scores = []
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=manual_seed)

    lr_oof = np.zeros((len(samples) - infer_size, 2))
    lr_predictions = np.zeros((infer_size, 2))

    for index, (train_index, valid_index) in enumerate(kf.split(X=train_data, y=labels)):
        logger.info(f"Fold {index + 1}")
        x_train, label_train = train_data[train_index], labels[train_index]
        x_valid, label_valid = train_data[valid_index], labels[valid_index]

        model = LogisticRegression(C=1, n_jobs=20)
        model.fit(X=x_train, y=label_train)

        lr_oof[valid_index] = model.predict_proba(x_valid)
        score = roc_auc_score(label_valid, lr_oof[valid_index][:, 1])
        logger.info(f"Fold {index + 1} auc：{score}")
        scores.append(score)

        lr_predictions += model.predict_proba(infer_data) / k_fold

    logger.info(f"Final auc：{np.mean(scores)}")
    return lr_predictions


def inferences(file_path: str,
               train_file_path: str,
               result_file_path: str,
               ngram_range: tuple = (1, 5),
               k_fold: int = 5,
               delimiter: str = "\t",
               manual_seed: int = 1) -> NoReturn:
    """
    :param file_path: 推理数据集集路径
    :param train_file_path: 训练数据集文件路径
    :param result_file_path: 结果保存路径
    :param ngram_range: 词袋范围
    :param k_fold: k折
    :param delimiter: 分隔符
    :param manual_seed: 随机种子
    """
    df_train = pd.read_table(filepath_or_buffer=train_file_path,
                             names=["text_a", "text_b", "label"], delimiter=delimiter).fillna("0")
    df_test = pd.read_table(filepath_or_buffer=file_path,
                            names=["text_a", "text_b"], delimiter=delimiter).fillna("0")

    labels = df_train["label"].values
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["text"] = df["text_a"] + " " + df["text_b"]

    res = infer(df["text"], labels, len(df_test), ngram_range, k_fold, manual_seed)
    pd.DataFrame(res[:, 1]).to_csv(result_file_path, index=False, header=False)


def inference(query1: str,
              query2: str,
              train_file_path: str,
              delimiter: str = "\t",
              ngram_range: tuple = (1, 5),
              k_fold: int = 5,
              manual_seed: int = 1) -> np.ndarray:
    """
    :param query1: 文本1
    :param query2: 文本2
    :param train_file_path: 训练数据集文件路径
    :param delimiter: 分隔符
    :param ngram_range: 词袋范围
    :param k_fold: k折
    :param manual_seed: 随机种子
    """
    samples, labels = list(), list()
    with open(train_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n")
            line = line.split(delimiter)

            if not line or len(line) != 3:
                continue

            text_a, text_b, label = line[0], line[1], line[2]
            samples.append(f"{text_a} {text_b}")
            labels.append(label)

        samples.append(f"{query1} {query2}")
        res = infer(samples, labels, 1, ngram_range, k_fold, manual_seed)

        return res[:, 1]


def actuator(train_file_path: str,
             query1: str = None,
             query2: str = None,
             file_path: str = None,
             result_file_path: str = None,
             ngram_range: tuple = (1, 5),
             k_fold: int = 5,
             delimiter: str = "\t",
             manual_seed: int = 1) -> Any:
    """
    :param train_file_path: 训练数据集文件路径
    :param query1: 文本1
    :param query2: 文本2
    :param file_path: 推理数据文件，pair不传
    :param result_file_path: 结果保存路径，pair不传
    :param ngram_range: 词袋范围
    :param k_fold: k折
    :param delimiter: 分隔符
    :param manual_seed: 随机种子
    """

    # 这里在日志文件里面做一个执行分割
    key = int(datetime.now().timestamp())
    logger.info("========================{}========================".format(key))

    if query1 and query2:
        return inference(query1, query2, train_file_path, delimiter, ngram_range, k_fold, manual_seed)
    elif file_path:
        inferences(file_path, train_file_path, result_file_path, ngram_range, k_fold, delimiter, manual_seed)


if __name__ == '__main__':
    actuator("./corpus/chinese/breeno/train.tsv", query1="12 23 4160 276", query2="29 23 169 1495")
