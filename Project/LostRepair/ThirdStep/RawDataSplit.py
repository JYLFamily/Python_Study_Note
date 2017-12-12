# coding:utf-8

import pandas as pd
from sklearn.model_selection import train_test_split


class RawDataSplit(object):

    def __init__(self, input_path, sep=",", test_size=0.2, random_state=9):
        self.__input_path = input_path
        self.__sep = sep
        self.__test_size = test_size
        self.__random_state = random_state
        self.__X = None
        self.__y = None
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None

    def set_train_test(self):
        self.__X = pd.read_csv(self.__input_path, sep=self.__sep, usecols=list(range(1, 53)))
        self.__y = pd.read_csv(self.__input_path, sep=self.__sep, usecols=[0])
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

    def get_train_test(self):
        # DataFrame shape (m, 1) -> series
        return (self.__train.values,
                self.__train_label.squeeze().values,
                self.__test.values,
                self.__test_label.squeeze().values)
