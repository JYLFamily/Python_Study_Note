# coding:utf-8

import re
import numpy as np


class ModelDataAgg(object):

    def __init__(self, train, test, *shape, **data_set):
        self.__train = train
        self.__test = test
        self.__train_rows = shape[0]
        self.__train_cols = int(len(data_set)/2)
        self.__test_rows = shape[1]
        self.__test_cols = int(len(data_set)/2)
        self.__data_set = data_set
        self.__train_set = dict()
        self.__test_set = dict()
        self.__train_oof = np.zeros((self.__train_rows, self.__train_cols))
        self.__test_oof = np.zeros((self.__test_rows, self.__test_cols))
        self.__train_all = None
        self.__test_all = None

    def train_test_split(self):
        for key, value in self.__data_set.items():
            if re.search(r"train", key):
                self.__train_set[key] = value
            else:
                self.__test_set[key] = value

    def train_merge(self):
        for i, (key, value) in enumerate(self.__train_set.items()):
            self.__train_oof[:, i] = value.reshape((-1, ))
        self.__train_all = np.hstack((self.__train, self.__train_oof))

        return self.__train_all

    def test_merge(self):
        for i, (key, value) in enumerate(self.__test_set.items()):
            self.__test_oof[:, i] = value.reshape((-1, ))
        self.__test_all = np.hstack((self.__test, self.__test_oof))

        return self.__test_all
