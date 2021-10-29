#! -*- coding: utf-8 -*-
""" Implementation of LSH
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from sim.base import LSH
from typing import Any


class MinHash(LSH):
    """ min-hash LSH实现
    """

    def __init__(self):
        super(MinHash, self).__init__()

    @staticmethod
    def gen_sig_vec(matrix: np.ndarray) -> list:
        """ 生成签名向量
        :param matrix: 必须是二维矩阵
        :return: 签名向量
        """
        count = 0
        seq_list = [i for i in range(matrix.shape[0])]
        result = [-1 for i in range(matrix.shape[1])]

        while len(seq_list) > 0:
            rand_seq = random.choice(seq_list)
            for i in range(matrix.shape[1]):
                if matrix[rand_seq][i] != 0 and result[i] == -1:
                    result[i] = rand_seq
                    count += 1

            if count == matrix.shape[1]:
                break

            seq_list.remove(rand_seq)

        return result

    def gen_sig_matrix(self, matrix: np.ndarray, n: int) -> np.ndarray:
        """ 生成签名向量matrix
        :param matrix: 必须是二维矩阵
        :param n: 签名向量matrix的行数
        :return: 签名向量matrix
        """
        result = list()

        for i in range(n):
            result.append(self.gen_sig_vec(matrix))

        return np.array(result)

    def min_hash(self, matrix: np.ndarray, band: int = 20, row: int = 5, hash_obj: str = "md5") -> dict:
        """ min-hash核心代码
        :param matrix: 必须是二维矩阵
        :param band: 将矩阵划分为bank的数量
        :param row: t每个block的行数
        :param hash_obj: 计算hash值的实现类
        :return: hash bucket字典，其中key是hash-val，value是行数(候选集索引)
        """
        hash_bucket = dict()
        # 决定了在矩阵上重排列的次数
        n = band * row
        sig_matrix = self.gen_sig_matrix(matrix, n)
        begin, end, count = 0, row, 0

        while end <= sig_matrix.shape[0]:
            count += 1
            for col in range(sig_matrix.shape[1]):
                tag = self.hash(str(sig_matrix[begin: begin + row, col]) + str(count), hash_obj)

                if tag not in hash_bucket:
                    hash_bucket[tag] = [col]
                elif col not in hash_bucket[tag]:
                    hash_bucket[tag].append(col)

            begin += row
            end += row

        return hash_bucket

    def search(self, candidates: Any, query: Any, band: int = 20, row: int = 5, hash_obj: str = "md5") -> set:
        """ min-hash Match search
        :param candidates: 必须是二维等长list或numpy array
        :param query: 必须是一维list或numpy array, 长度与候选集长度相同
        :param band: 将矩阵划分为bank的数量
        :param row: t每个block的行数
        :param hash_obj: 计算hash值的实现类
        :return: search结果集合
        """
        if not isinstance(candidates, (list, np.ndarray)):
            raise TypeError("must be list or numpy array")

        result = set()
        matrix = None

        if isinstance(candidates, list):
            candidates.append(query)
            matrix = np.array(candidates).T
        elif isinstance(candidates, np.ndarray):
            if isinstance(query, list):
                query = np.array([query])
            elif isinstance(query, np.ndarray):
                query = query[np.newaxis, :]
            matrix = np.concatenate((candidates, query), axis=0).T

        hash_bucket = self.min_hash(matrix=matrix, band=band, row=row, hash_obj=hash_obj)
        query_col = matrix.shape[1] - 1
        for key in hash_bucket:
            if query_col in hash_bucket[key]:
                for i in hash_bucket[key]:
                    result.add(i)

        result.remove(query_col)
        return result


class TableNode(object):
    """ Hash table的节点
    """

    def __init__(self, index):
        self.val = index
        self.buckets = {}


class E2LSH(LSH):
    """ P-table LSH实现类
    """

    def __init__(self):
        super(E2LSH, self).__init__()

    @staticmethod
    def gen_para(length: int, r: int = 1) -> tuple:
        """ 生成para分布
        :param length: 向量长度
        :param r: 阈值
        :return: para分布向量
        """
        gauss = list()
        for i in range(length):
            gauss.append(random.gauss(0, 1))
        uniform = random.uniform(0, r)

        return gauss, uniform

    def gen_e2lsh_family(self, length: int, k: int = 20, r: int = 1) -> list:
        """ 生成p-table的lsh cluster
        :param length: 向量长度
        :param k: 生成的数量
        :param r: 阈值
        :return: 分布cluster
        """
        result = list()
        for i in range(k):
            result.append(self.gen_para(length, r))

        return result

    @staticmethod
    def gen_hash_values(e2lsh_family: list, vector: int, r: int = 1) -> list:
        """ 计算hash值
        :param e2lsh_family: 分布cluster
        :param vector:
        :param r: 可调参数
        :return: hash value列表
        """
        hash_values = list()

        for hab in e2lsh_family:
            hash_value = (np.inner(hab[0], vector) + hab[1]) // r
            hash_values.append(hash_value)

        return hash_values

    @staticmethod
    def h2(hash_values: list, fp_rand: list, c: int = pow(2, 32) - 5, k: int = 20) -> int:
        """ 计算fingerprint值
        :param hash_values: hash value列表
        :param fp_rand: 一组用于生成fingerprints的随机数
        :param k: 生成数量
        :param c: 可调参数
        :return: fingerprint值，int类型
        """
        return int(sum([(hash_values[i] * fp_rand[i]) for i in range(k)]) % c)

    def e2lsh(self, candidates: Any, c: int = pow(2, 32) - 5, k: int = 20,
              L: int = 5, r: int = 1, table_size: int = 20) -> tuple:
        """ E2LSH的核心实现
        :param candidates: 必须是二维等长列表
        :param c: 可调参数
        :param k: 生成数量
        :param L: 可调参数
        :param r: 可调参数
        :param table_size: hash table大小
        :return: hash table,hash func,fp_rand(一组用于生成fingerprints的随机数)
        """
        hash_table = [TableNode(i) for i in range(table_size)]

        n = len(candidates[0])
        m = len(candidates)
        hash_funcs = list()
        fp_rand = [random.randint(-10, 10) for i in range(k)]

        for times in range(L):
            e2lsh_family = self.gen_e2lsh_family(n, k, r)
            hash_funcs.append(e2lsh_family)

            for data_index in range(m):
                hash_values = self.gen_hash_values(e2lsh_family, candidates[data_index], r)
                fp = self.h2(hash_values, fp_rand, c, k)
                index = fp % table_size
                node = hash_table[index]

                if fp in node.buckets:
                    bucket = node.buckets[fp]
                    bucket.append(data_index)
                else:
                    node.buckets[fp] = [data_index]

        return hash_table, hash_funcs, fp_rand

    def search(self, candidates: Any, query: Any, c: int = pow(2, 32) - 5,
               k: int = 20, L: int = 5, r: int = 1, table_size: int = 20) -> set:
        """ min-hash Match search
        :param candidates: 必须是二维等长列表
        :param query: 必须是一维list或numpy array, 长度与候选集长度相同
        :param c: 可调参数
        :param k: 生成数量
        :param L: 可调参数
        :param r: 可调参数
        :param table_size: hash table大小
        :return: search结果集合
        """
        if not isinstance(candidates, list) and not isinstance(candidates[0], list):
            raise TypeError("must be 2-d list")

        result = set()
        hash_table, hash_funcs, fp_rand = self.e2lsh(candidates, c, k, L, r, table_size)

        for hash_func in hash_funcs:
            query_fp = self.h2(self.gen_hash_values(hash_func, query, r), fp_rand, c, k)
            query_index = query_fp % table_size

            if query_fp in hash_table[query_index].buckets:
                result.update(hash_table[query_index].buckets[query_fp])

        return result
