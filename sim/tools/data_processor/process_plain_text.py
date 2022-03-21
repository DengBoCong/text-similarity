#! -*- coding: utf-8 -*-
""" process plain text
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import random
import networkx as nx
import os
import pandas as pd
import pickle as pkl
import uuid
from sim.tools.settings import RUNTIME_LOG_FILE_PATH
from sim.tools.tokenizer import pad_sequences
from sim.tools.tokenizer import Segment
from sim.tools.tokenizer import BertTokenizer as CustomBertTokenizer
from sim.tools.tokenizer import Tokenizer
from sim.tools.tools import get_logger
from tokenizers.implementations.bert_wordpiece import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer
from typing import NoReturn

logger = get_logger(name="processor", file_path=RUNTIME_LOG_FILE_PATH)


def text_pair_to_token_id(file_path: str,
                          save_path: str,
                          split: str = "\t",
                          seg_model: str = "jieba",
                          pad_max_len: int = None,
                          padding: str = 'post',
                          truncating: str = 'post',
                          value: int = 0,
                          print_count: int = 1000,
                          tokenizer: Tokenizer = None) -> Tokenizer:
    """ 将Text pair转换为token id
    :param file_path: 未处理的文本数据路径，文本格式: <text1><split><text2><split><label>
    :param save_path: 保存处理后的数据路径
    :param split: text pair的分隔符
    :param seg_model: 分词工具model，支持jieba, lac, pkuseg
    :param pad_max_len: padding size
    :param padding: 填充类型，pre在前，post在后
    :param truncating: 截断类型，pre在前，post在后
    :param value: 填充值类型，float或者是string
    :param print_count: 处理print_count数量数据打印日志
    :param tokenizer: 分词器
    :return: 分词器
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Raw text file not found")

    count, segment, text1s, text2s, labels = 0, None, list(), list(), list()

    if seg_model:
        segment = Segment(model=seg_model)
    with open(file_path, "r", encoding="utf-8") as raw_file, open(
            save_path, "w", encoding="utf-8") as save_file:
        for line in raw_file:
            line = line.strip().strip("\n")
            if line == "":
                continue

            pair = line.split(split)
            if seg_model:
                pair[0] = segment.cut(pair[0])
                pair[1] = segment.cut(pair[1])
            text1s.append(pair[0])
            text2s.append(pair[1])
            labels.append(pair[2] if len(pair) == 3 else 0)

            count += 1
            if count % print_count == 0:
                print("\r{} text-pairs processed".format(count), end="", flush=True)

        logger.info("{} text-pairs processed".format(count))

        if not tokenizer:
            tokenizer = Tokenizer(oov_token="[UNK]")
            tokenizer.fit_on_texts(texts=text1s + text2s)

        text1s = tokenizer.texts_to_sequences(texts=text1s)
        text2s = tokenizer.texts_to_sequences(texts=text2s)

        text1s = pad_sequences(sequences=text1s, max_len=pad_max_len,
                               padding=padding, truncating=truncating, value=value)
        text2s = pad_sequences(sequences=text2s, max_len=pad_max_len,
                               padding=padding, truncating=truncating, value=value)

        logger.info("Begin write in")
        for index, (text1, text2, label) in enumerate(zip(text1s, text2s, labels)):
            save_file.write(
                "{}{}{}{}{}\n".format(" ".join(map(str, text1)), split, " ".join(map(str, text2)), split, label))

            if index % print_count == 0:
                print("\r{} text-pairs processed".format(index), end="", flush=True)

        logger.info("Finish write in")

    return tokenizer


def text_to_token_id_for_bert(file_path: str,
                              save_path: str,
                              split: str = "\t",
                              pad_max_len: int = None,
                              tokenizer: BertTokenizer = None,
                              token_dict: str = None,
                              padding: str = 'post',
                              truncating: str = 'post',
                              value: int = 0,
                              is_single: bool = False,
                              print_count: int = 1000) -> NoReturn:
    """用于bert将Text转换为token id
    :param file_path: 未处理的文本数据路径，文本格式: <text1><split><text2><split><label>
    :param save_path: 保存处理后的数据路径
    :param split: text pair的分隔符
    :param pad_max_len: padding size
    :param tokenizer: 分词器
    :param token_dict: 映射字典或其文件路径
    :param padding: 填充类型，pre在前，post在后
    :param truncating: 截断类型，pre在前，post在后
    :param value: 填充值类型，float或者是string
    :param is_single: 是否处理成单条文本
    :param print_count: 处理print_count数量数据打印日志
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Raw text file not found")

    if not tokenizer:
        tokenizer = CustomBertTokenizer(token_dict=token_dict, do_lower_case=True)

    batch_token_ids, batch_segment_ids, batch_labels, count = [], [], [], 0
    with open(file_path, "r", encoding="utf-8") as raw_file, open(
            save_path, "w", encoding="utf-8") as save_file:
        for line in raw_file:
            line = line.strip().strip("\n")
            if line == "":
                continue

            pair = line.split(split)
            if is_single:
                a_token_ids, a_segment_ids = tokenizer.encode(first_text=pair[0], max_len=pad_max_len)
                batch_token_ids.append(a_token_ids)
                batch_segment_ids.append(a_segment_ids)
                b_token_ids, b_segment_ids = tokenizer.encode(first_text=pair[1], max_len=pad_max_len)
                batch_token_ids.append(b_token_ids)
                batch_segment_ids.append(b_segment_ids)
                batch_labels.append(pair[2] if len(pair) == 3 else 0)
                batch_labels.append(pair[2] if len(pair) == 3 else 0)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=pair[0], second_text=pair[1], max_len=pad_max_len)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(pair[2] if len(pair) == 3 else 0)

            count += 1
            if count % print_count == 0:
                print("\r{} text-pairs processed".format(count), end="", flush=True)

        logger.info("{} text-pairs processed".format(count))

        batch_token_ids = pad_sequences(sequences=batch_token_ids, max_len=pad_max_len,
                                        padding=padding, truncating=truncating, value=value)
        batch_segment_ids = pad_sequences(sequences=batch_segment_ids, max_len=pad_max_len,
                                          padding=padding, truncating=truncating, value=value)

        logger.info("Begin write in")
        for index, (token_ids, segment_ids, labels) in enumerate(zip(batch_token_ids, batch_segment_ids, batch_labels)):
            save_file.write("{}{}{}{}{}\n".format(" ".join(map(str, token_ids)),
                                                  split, " ".join(map(str, segment_ids)), split, labels))

            if index % print_count == 0:
                print("\r{} text-pairs processed".format(index), end="", flush=True)

        logger.info("Finish write in")


def tetrad_text_to_token_id_for_bert(file_path: str,
                                     save_path: str,
                                     split: str = "\t",
                                     pad_max_len: int = None,
                                     tokenizer: BertTokenizer = None,
                                     token_dict: str = None,
                                     padding: str = 'post',
                                     truncating: str = 'post',
                                     value: int = 0,
                                     print_count: int = 1000) -> NoReturn:
    """用于bert将Text转换为token id, 四个文本一组
    :param file_path: 未处理的文本数据路径，文本格式: <text1><split><text2><split><text3><split><text4><split><label>
    :param save_path: 保存处理后的数据路径
    :param split: text pair的分隔符
    :param pad_max_len: padding size
    :param tokenizer: 分词器
    :param token_dict: 映射字典或其文件路径
    :param padding: 填充类型，pre在前，post在后
    :param truncating: 截断类型，pre在前，post在后
    :param value: 填充值类型，float或者是string
    :param print_count: 处理print_count数量数据打印日志
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Raw text file not found")

    if not tokenizer:
        tokenizer = CustomBertTokenizer(token_dict=token_dict, do_lower_case=True)

    batch_token_ids_a, batch_token_ids_b, batch_labels, count = [], [], [], 0
    batch_segment_ids_a, batch_segment_ids_b = [], []
    with open(file_path, "r", encoding="utf-8") as raw_file, open(save_path, "w", encoding="utf-8") as save_file:
        for line in raw_file:
            line = line.strip().strip("\n")
            if line == "":
                continue

            pair = line.split(split)

            a_token_ids, a_segment_ids = tokenizer.encode(first_text=pair[0], max_len=pad_max_len)
            batch_token_ids_a.append(a_token_ids)
            batch_segment_ids_a.append(a_segment_ids)
            b_token_ids, b_segment_ids = tokenizer.encode(first_text=pair[1], max_len=pad_max_len)
            batch_token_ids_b.append(b_token_ids)
            batch_segment_ids_b.append(b_segment_ids)
            batch_labels.append(pair[2] if len(pair) == 3 else 0)

            count += 1
            if count % print_count == 0:
                print("\r{} text-pairs processed".format(count), end="", flush=True)

        logger.info("{} text-pairs processed".format(count))

        batch_token_ids_a = pad_sequences(sequences=batch_token_ids_a, max_len=pad_max_len,
                                          padding=padding, truncating=truncating, value=value)
        batch_token_ids_b = pad_sequences(sequences=batch_token_ids_b, max_len=pad_max_len,
                                          padding=padding, truncating=truncating, value=value)
        batch_segment_ids_a = pad_sequences(sequences=batch_segment_ids_a, max_len=pad_max_len,
                                            padding=padding, truncating=truncating, value=value)
        batch_segment_ids_b = pad_sequences(sequences=batch_segment_ids_b, max_len=pad_max_len,
                                            padding=padding, truncating=truncating, value=value)

        logger.info("Begin write in")
        for index, (token_ids_a, segment_ids_a, token_ids_b, segment_ids_b, labels) in enumerate(
                zip(batch_token_ids_a, batch_segment_ids_a, batch_token_ids_b, batch_segment_ids_b, batch_labels)
        ):
            save_file.write("{}{}{}{}{}{}{}{}{}\n".format(
                " ".join(map(str, token_ids_a)), split, " ".join(map(str, segment_ids_a)), split,
                " ".join(map(str, token_ids_b)), split, " ".join(map(str, segment_ids_b)), split, labels
            ))

            if index % print_count == 0:
                print("\r{} text-pairs processed".format(index), end="", flush=True)

        logger.info("Finish write in")


def convert_sample_to_json(file_path: str,
                           save_path: str,
                           header: str = None,
                           delimiter: str = "\t") -> NoReturn:
    """将单个样本构造成(rid, labels, text_a, text_b)的json形式
    :param file_path: 数据集文件路径
    :param save_path: json文本保存路径
    :param header: 文件是否有列头
    :param delimiter: 句间分隔符
    """
    input_df = pd.read_csv(filepath_or_buffer=file_path, header=header, delimiter=delimiter)
    records = []
    for _, record in tqdm(input_df.iterrows()):
        if len(record) > 2:
            labels = record[2]
        else:
            labels = 2
        records.append({
            "rid": uuid.uuid4().hex,
            "labels": labels,
            "text_a": str(record[0]),
            "text_b": str(record[1])
        })

    with open(save_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(f"{json.dumps(record, ensure_ascii=False)}\n")


def construct_meta_info(file_path: str, save_path: str) -> NoReturn:
    """构建数据集样本的元信息及索引
    :param file_path: 数据集json文件路径
    :param save_path: meta info文件保存路径
    """
    meta_info = {
        "sid2sentence": {},
        "sentence2sid": {},
        "sentence_graph": nx.Graph()
    }

    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            if line.strip().strip("\n") == "":
                continue
            record = json.loads(line)
            sentences = [record["text_a"], record["text_b"]]
            for sentence in sentences:
                if sentence not in meta_info["sentence2sid"]:
                    sid = uuid.uuid4().hex
                    meta_info["sid2sentence"][sid] = sentence
                    meta_info["sentence2sid"][sentence] = sid
                    meta_info["sentence_graph"].add_node(sid)
            if record["labels"] != 2:
                meta_info["sentence_graph"].add_edge(meta_info["sentence2sid"][sentences[0]],
                                                     meta_info["sentence2sid"][sentences[1]])

    with open(save_path, "w", encoding="utf-8") as file:
        pkl.dump(meta_info, file)


def construct_enhanced_data(file_path_list: list, save_path: str, manual_seed: int = 1) -> NoReturn:
    """构建闭包数据
    :param file_path_list: 数据集json文件路径列表
    :param save_path: enhanced文件保存路径
    :param manual_seed: 随机种子
    """
    random.seed(manual_seed)
    enhanced_records = []
    text_pair_set = set()

    for file_path in file_path_list:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in tqdm(file):
                record = json.loads(line)
                if (record['text_a'], record['text_b']) in text_pair_set:
                    continue

                enhanced_records.append(record)
                record_dup = copy.deepcopy(record)
                record_dup["text_a"], record_dup["text_b"] = record["text_b"], record["text_a"]
                enhanced_records.append(record_dup)
    random.shuffle(enhanced_records)

    with open(save_path, "w", encoding="utf-8") as file:
        for record in tqdm(enhanced_records):
            file.write(f"{json.dumps(record, ensure_ascii=False)}\n")


def get_k_fold_data(file_path: str,
                    meta_info_file_path: str,
                    save_dir: str,
                    train_file: str = "train.jsonl",
                    dev_file: str = "dev.jsonl",
                    enhanced_file: str = "train_enhanced.jsonl",
                    n_splits: int = 5,
                    manual_seed: int = 1) -> NoReturn:
    """拆分k折数据集
    :param file_path: 数据集json文件路径
    :param meta_info_file_path: meta info文件路径
    :param save_dir: k折数据集保存目录
    :param train_file: 训练集文件
    :param dev_file: 验证集文件
    :param enhanced_file: 闭包训练集文件
    :param n_splits: k折
    :param manual_seed: 随机种子
    """
    random.seed(manual_seed)
    # 获取所有连通图
    with open(meta_info_file_path, "rb", encoding="utf-8") as file:
        meta_info = pkl.load(file)
    logger.info('compute connected components')
    components = list(nx.connected_components(meta_info["sentence_graph"]))
    logger.info(f'components number: {len(components)}')
    sid2cid = {}
    for cid, component in enumerate(components):
        for sid in component:
            sid2cid[sid] = cid

    records_components = [[] for _ in range(len(components))]
    logger.info('compute records components')
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            if line.strip().strip("\n") == "":
                continue
            record = json.loads(line)
            sid_a, sid_b = meta_info["sentence2sid"][record["text_a"]], meta_info["sentence2sid"][record["text_b"]]
            assert sid2cid[sid_a] == sid2cid[sid_b], "cid_a != cid_b"
            records_components[sid2cid[sid_a]].append(record)
    random.shuffle(records_components)

    # 生成k fold data
    logger.info('generate k fold data')
    per_split_num = int(len(records_components) / n_splits)
    for split_id in tqdm(range(n_splits)):
        os.mkdir(os.path.join(save_dir, str(split_id)))
        dev_ids = list(range(split_id * per_split_num, (split_id + 1) * per_split_num))
        train_records, dev_records = [], []
        for idx, records_component in enumerate(records_components):
            if idx in dev_ids:
                dev_records.extend(records_component)
            else:
                train_records.extend(records_component)
        random.shuffle(train_records)

        with open(os.path.join(save_dir, str(split_id), train_file), "w", encoding="utf-8") as file:
            for record in tqdm(train_records):
                file.write(f"{json.dumps(record, ensure_ascii=False)}\n")
        with open(os.path.join(save_dir, str(split_id), dev_file), "w", encoding="utf-8") as file:
            for record in tqdm(dev_records):
                file.write(f"{json.dumps(record, ensure_ascii=False)}\n")

        # generate enhanced train data
        construct_enhanced_data(file_path_list=[os.path.join(save_dir, str(split_id), train_file)],
                                save_path=os.path.join(save_dir, str(split_id), enhanced_file), manual_seed=manual_seed)


def construct_tokenizer_data_for_record(file_path: str, save_path: str):
    """用于给record数据格式的数据构建tokenizer使用是数据
    :param file_path: 数据集json文件路径
    :param save_path: enhanced文件保存路径
    """
    with open(file_path, 'w') as data_file:
        with open(save_path) as save_file:
            for line in tqdm(save_file):
                record = json.loads(line)
                data_file.write(f"{record['text_a']}\n")
                data_file.write(f"{record['text_b']}\n")


def train_tokenizer(tokenizer_file_path: str,
                    model_name: str,
                    vocab_size: int = 22000,
                    manual_seed: int = 1) -> NoReturn:
    """生成tokenizer配置文件
    :param tokenizer_file_path: 用于分词的文件路径，一行一个text
    :param model_name: 保存tokenizer文件名或者目录路径
    :param vocab_size: 词表大小
    :param manual_seed: 随机种子
    """
    random.seed(manual_seed)
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokenizer.train(
        files=[tokenizer_file_path],
        vocab_size=vocab_size,
        min_frequency=0,
        special_tokens=special_tokens,
        limit_alphabet=vocab_size,
        wordpieces_prefix="##"
    )
    os.makedirs(model_name, exist_ok=True)
    tokenizer.save_model(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                              do_lower_case=False,
                                              strip_accents=False)
    tokenizer.save_pretrained(model_name)
    logger.info(f'save tokenizer, with vocab_size: {tokenizer.vocab_size}')
