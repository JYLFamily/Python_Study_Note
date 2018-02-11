# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ReadData(object):
    def __init__(self, *, input_path, header, sep, feature_index, target_index, test_size, random_state):
        self.__input_path = input_path
        self.__header = header
        self.__sep = sep
        self.__test_size = test_size
        self.__random_state = random_state
        self.__X = pd.read_csv(self.__input_path, header=self.__header, sep=self.__sep, usecols=feature_index)
        self.__y = pd.read_csv(self.__input_path, header=self.__header, sep=self.__sep, usecols=target_index)
        self.__train, self.__test, self.__train_label, self.__test_label = [None for _ in range(4)]

    def get_train_test(self):
        self.__X = self.__X.loc[(np.logical_not(self.__y is np.nan)), :]
        self.__y = self.__y[np.logical_not(self.__y is np.nan)]

        self.__train, self.__test, self.__train_label, self.__test_label = train_test_split(
            self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state
        )

        return self.__train, self.__test, self.__train_label, self.__test_label