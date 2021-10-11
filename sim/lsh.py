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
    """ Implementation of min-hash LSH
    """

    def __init__(self):
        super(MinHash, self).__init__()

    @staticmethod
    def gen_sig_vec(matrix: np.ndarray) -> list:
        """ generate signature vector
        :param matrix: must be a two-dimensional matrix
        :return: signature vector
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
        """ generate signature vector matrix
        :param matrix: must be a two-dimensional matrix
        :param n: the rows of signature vector matrix
        :return: signature vector matrix
        """
        result = list()

        for i in range(n):
            result.append(self.gen_sig_vec(matrix))

        return np.array(result)

    def min_hash(self, matrix: np.ndarray, band: int = 20, row: int = 5, hash_obj: str = "md5") -> dict:
        """ the core of min-hash
        :param matrix: must be a two-dimensional matrix
        :param band: divide the matrix into band blocks
        :param row: the size of each block is row
        :param hash_obj: the method used to calculate the hash
        :return: a hash bucket dict, where key is the hash-val and value is the number of columns (candidate set index)
        """
        hash_bucket = dict()
        # This determines how many permutations need to be performed on the matrix
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
        :param candidates: must be a two-dimensional equal-length list or numpy array
        :param query: must be a one-dimensional list or numpy array, the length is the same as the length of candidates
        :param band: divide the matrix into band blocks
        :param row: the size of each block is row
        :param hash_obj: the method used to calculate the hash
        :return: search result collection
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
    """ The node of Hash table
    """

    def __init__(self, index):
        self.val = index
        self.buckets = {}


class E2LSH(LSH):
    """ Implementation of P-table LSH
    """

    def __init__(self):
        super(E2LSH, self).__init__()

    @staticmethod
    def gen_para(length: int, r: int = 1) -> tuple:
        """ generate para distribution
        :param length: vector length
        :param r: threshold
        :return: para distribution vector
        """
        gauss = list()
        for i in range(length):
            gauss.append(random.gauss(0, 1))
        uniform = random.uniform(0, r)

        return gauss, uniform

    def gen_e2lsh_family(self, length: int, k: int = 20, r: int = 1) -> list:
        """ generate p-table lsh cluster
        :param length: vector length
        :param k: num of generations
        :param r: threshold
        :return: distribution cluster
        """
        result = list()
        for i in range(k):
            result.append(self.gen_para(length, r))

        return result

    @staticmethod
    def gen_hash_values(e2lsh_family: list, vector: int, r: int = 1) -> list:
        """ Calculate the hash value
        :param e2lsh_family: distribution cluster
        :param vector:
        :param r: adjustable factor
        :return: hash value list
        """
        hash_values = list()

        for hab in e2lsh_family:
            hash_value = (np.inner(hab[0], vector) + hab[1]) // r
            hash_values.append(hash_value)

        return hash_values

    @staticmethod
    def h2(hash_values: list, fp_rand: list, c: int = pow(2, 32) - 5, k: int = 20) -> int:
        """ Calculate the fingerprint value
        :param hash_values: hash value list
        :param fp_rand: a set of random values used to generate fingerprints
        :param k: num of generations
        :param c: adjustable factor
        :return: fingerprint value, int
        """
        return int(sum([(hash_values[i] * fp_rand[i]) for i in range(k)]) % c)

    def e2lsh(self, candidates: Any, c: int = pow(2, 32) - 5, k: int = 20,
              L: int = 5, r: int = 1, table_size: int = 20) -> tuple:
        """ the core of E2LSH
        :param candidates: must be a two-dimensional equal-length list
        :param c: adjustable factor
        :param k: num of generations
        :param L: adjustable factor
        :param r: adjustable factor
        :param table_size: hash table size
        :return: return hash table, hash func, fp_rand(a set of random values used to generate fingerprints)
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
        :param candidates: must be a two-dimensional equal-length list
        :param query: must be a one-dimensional list or numpy array, the length is the same as the length of candidates
        :param c: adjustable factor
        :param k: num of generations
        :param L: adjustable factor
        :param r: adjustable factor
        :param table_size: hash table size
        :return: search result collection
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
