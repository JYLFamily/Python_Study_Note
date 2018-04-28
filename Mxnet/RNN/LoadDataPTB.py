# coding:utf-8

import os
import time
import math
import zipfile
import numpy as np
from mxnet import gluon, autograd, nd


class Dictionary(object):
    def __init__(self):
        # 字典
        self.word_to_idx = {}
        # 去重字符序列
        self.idx_to_word = []

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.idx_to_word.append(word)
            self.word_to_idx[word] = len(self.idx_to_word) - 1

    def __len__(self):
        return len(self.idx_to_word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "ptb.train.txt"))
        self.valid = self.tokenize(os.path.join(path, "ptb.valid.txt"))
        self.test = self.tokenize(os.path.join(path, "ptb.test.txt"))

    def tokenize(self, path):
        """这里有个小问题, 有可能建立 train, valid, test 的字典是不相同的（实际查看后是相同的）"""
        assert os.path.exists(path)
        # 将词语添加至词典
        with open(path, "r") as f:
            # 文件中共有多少个字符
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # 将文本转换成词语索引的序列
        with open(path, "r") as f:
            indices = nd.zeros((tokens,), dtype="int32")
            idx = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    indices[idx] = self.dictionary.word_to_idx[word]
                    idx += 1
        return indices


if __name__ == "__main__":
    with zipfile.ZipFile("D:\\Code\\Python\\Python_Study_Note\\Mxnet\\RNN\\ptb.zip", "r") as zin:
        zin.extractall("D:\\Code\\Python\\Python_Study_Note\\Mxnet\\RNN")

    corpus = Corpus("D:\\Code\\Python\\Python_Study_Note\\Mxnet\\RNN\\ptb")
    vocab_size = len(corpus.dictionary)
    print(vocab_size)

