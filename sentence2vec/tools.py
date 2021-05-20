#! -*- coding: utf-8 -*-
""" Coding Tools
"""


# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License


# 进行词频统计
def counter(sentences):
    word_counts = []
    for sentence in sentences:
        count = {}
        for word in sentence:
            if not count.get(word):
                count.update({word: 1})
            elif count.get(word):
                count[word] += 1
        word_counts.append(count)
    return word_counts


# 计算TF(word代表被计算的单词，word_list是被计算单词所在文档分词后的字典)
def tf(word, word_list):
    return word_list.get(word) / sum(word_list.values())


# 统计含有该单词的句子数
def count_sentence(word, wordcount):
    return sum(1 for i in wordcount if i.get(word))


# 计算IDF
def idf(word, wordcount):
    return math.log(len(wordcount) / (count_sentence(word, wordcount) + 1))


# 计算TF-IDF
def tfidf(word, word_list, wordcount):
    return tf(word, word_list) * idf(word, wordcount)
