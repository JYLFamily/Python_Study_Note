# coding:utf-8

import numpy as np
from sklearn.model_selection import StratifiedKFold


class OutOfFold(object):

    def __init__(self, clf, train, train_label, test, cv=5, random_state=9):
        self.__clf = clf
        self.__train = train
        self.__train_label = train_label
        self.__test = test
        self.__cv = cv
        self.__random_state = random_state
        self.__skf = StratifiedKFold()
        #
        self.__oof_train = None
        self.__oof_test = None
        self.__oof_test_skf = None
        #
        self.__x_train = None
        self.__y_train = None
        self.__x_test = None

    def set_skf(self):
        self.__skf = StratifiedKFold(n_splits=self.__cv,
                                     shuffle=True,
                                     random_state=self.__random_state)

    def get_oof(self):
        self.__oof_train = np.zeros((self.__train.shape[0], 1))
        self.__oof_test = np.zeros((self.__test.shape[0], 1))
        self.__oof_test_skf = np.empty((self.__test.shape[0], self.__cv))

        for i, (train_index, test_index) in enumerate(self.__skf.split(self.__train, self.__train_label)):
            self.__x_train = self.__train[train_index]
            self.__y_train = self.__train_label[train_index]
            self.__x_test = self.__train[test_index]

            self.__clf.train(self.__x_train, self.__y_train)

            self.__oof_train[test_index] = self.__clf.predict(self.__x_test)
            self.__oof_test_skf[:, i] = self.__clf.predict(self.__x_test)

        self.__oof_test = self.__oof_test_skf.mean(axis=1)

        return self.__oof_train, self.__oof_test