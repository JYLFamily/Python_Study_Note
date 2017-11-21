# coding:utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from Project.LostRepair.OutOfFold import OutOfFold
from Project.LostRepair.ModelAssistant import ModelAssistant


class ThirdStep(object):

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
        self.__X = pd.read_csv(self.__input_path, sep=self.__sep, usecols=list(range(1, 34)))
        self.__y = pd.read_csv(self.__input_path, sep=self.__sep, usecols=[0])
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

    def get_train(self):
        return self.__train.values

    def get_train_label(self):
        # 一列 DaTaFrame 变为 Series
        # 再变为 ndarray
        return self.__train_label.squeeze().values

    def get_test(self):
        return self.__test.values

    def get_test_label(self):
        # 先一列 DaTaFrame 变为 Series
        # 再变为 ndarray
        return self.__test_label.squeeze().values


if __name__ == "__main__":
    # get train and train_label and test and test_label
    ts = ThirdStep(input_path="C:\\Users\\Dell\\Desktop\\zytsl_robot.csv")
    ts.set_train_test()
    train = ts.get_train()
    train_label = ts.get_train_label()
    test = ts.get_test()
    test_label = ts.get_test_label()

    # set clf
    param = {
        "learning_rate": 0.05,
        "n_estimators": 50,
        "subsample": 0.6,
        "max_depth": 7,
        "min_samples_split": 800,
        "min_samples_leaf": 50,
        "max_features": "sqrt"
    }
    gbc_ma = ModelAssistant(clf=GradientBoostingClassifier, params=param)

    # get oof_train and oof_test
    oof = OutOfFold(clf=gbc_ma, train=train, train_label=train_label, test=test_label)
    oof.set_skf()
    oof_train_gbc, oof_test_gbc = oof.get_oof()
    print(oof_train_gbc[0:10])
    print(oof_test_gbc[0:10])