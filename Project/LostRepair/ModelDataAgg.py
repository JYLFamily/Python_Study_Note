# coding:utf-8

import re
import numpy as np


class ModelDataAgg(object):

    def __init__(self, *shape, **data_set):
        self.__train_rows = shape[0]
        self.__train_cols = len(data_set)
        self.__test_rows = shape[1]
        self.__test_cols = len(data_set)
        self.__data_set = data_set
        self.__train_set = dict()
        self.__test_set = dict()
        self.__train_oof = np.zeros((self.__train_rows, self.__train_cols))
        self.__test_oof = np.zeros((self.__test_rows, self.__test_cols))

    def train_test_split(self):
        for key, value in self.__data_set.items():
            if re.search(r"train", key):
                self.__train_set[key] = value
            else:
                self.__test_set[key] = value

    def train_merge(self):
        for i, (key, value) in enumerate(self.__train_set.items()):
            self.__train_oof[:, i] = value.reshape((-1, ))
        return self.__train_oof

    def test_merge(self):
        for i, (key, value) in enumerate(self.__test_set.items()):
            self.__test_oof[:, i] = value.reshape((-1, ))
        return self.__test_oof
