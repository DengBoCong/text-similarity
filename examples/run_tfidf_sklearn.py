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
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tools import get_logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = get_logger(name="actuator", file_path=RUNTIME_LOG_FILE_PATH)


def main(train_file_path: str, valid_file_path: str):
    """
    :param train_file_path: 训练数据集文件路径
    :param valid_file_path: 验证数据集文件路径
    """
    df_train = pd.read_table(filepath_or_buffer=train_file_path,
                             names=["text_a", "text_b", "label"]).fillna("0")
    df_test = pd.read_table(filepath_or_buffer=valid_file_path,
                            names=["text_a", "text_b"]).fillna("0")
    label = df_train["label"].values
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["text"] = df["text_a"] + " " + df["text_b"]

    tf_idf = TfidfVectorizer(ngram_range=(1, 5))
    tf_idf_feature = tf_idf.fit_transform(df["text"])
    svd_feature = TruncatedSVD(n_components=100).fit_transform(tf_idf_feature)
    train_df = tf_idf_feature[:len(df_train)]
    test_df = tf_idf_feature[len(df_train):]

    k_fold, scores = 5, []
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=2020)

    lr_oof = np.zeros((len(df_train), 2))
    lr_predictions = np.zeros((len(df_test), 2))

    for index, (train_index, valid_index) in enumerate(kf.split(X=train_df, y=label)):
        logger.info(f"Fold {index + 1}")
        x_train, label_train = train_df[train_index], label[train_index]
        x_valid, label_valid = train_df[valid_index], label[valid_index]

        model = LogisticRegression(C=1, n_jobs=20)
        model.fit(X=x_train, y=label_train)

        lr_oof[valid_index] = model.predict_proba(x_valid)
        scores.append(roc_auc_score(label_valid, lr_oof[valid_index][:, 1]))

        lr_predictions += model.predict_proba(test_df) / k_fold

    np.mean(scores)




