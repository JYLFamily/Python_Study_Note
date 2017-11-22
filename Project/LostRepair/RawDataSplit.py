# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from Project.LostRepair.OutOfFold import OutOfFold
from Project.LostRepair.ModelAssistant import ModelAssistant
from Project.LostRepair.ModelDataAgg import ModelDataAgg
from Project.LostRepair.XgbModel import XgbModel


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
    ######
    # get train and train_label and test and test_label
    rds = RawDataSplit(input_path="C:\\Users\\Dell\\Desktop\\zytsl_robot.csv")
    rds.set_train_test()
    train = rds.get_train()
    train_label = rds.get_train_label()
    test = rds.get_test()
    test_label = rds.get_test_label()

    ######
    # set clf
    # gbc_param = {
    #     "learning_rate": 0.05,
    #     "n_estimators": 50,
    #     "subsample": 0.6,
    #     "max_depth": 7,
    #     "min_samples_split": 800,
    #     "min_samples_leaf": 50,
    #     "max_features": "sqrt"
    # }
    gbc_ma = ModelAssistant(clf=GradientBoostingClassifier)
    rf_ma = ModelAssistant(clf=RandomForestClassifier)
    ab_ma = ModelAssistant(clf=AdaBoostClassifier)
    lr_ma = ModelAssistant(clf=LogisticRegression)
    et_ma = ModelAssistant(clf=ExtraTreeClassifier)
    kn_ma = ModelAssistant(clf=KNeighborsClassifier)

    ######
    # get train_oof and test_oof from gbc model
    gbc_oof = OutOfFold(clf=gbc_ma, train=train, train_label=train_label, test=test)
    gbc_oof.set_skf()
    gbc_train_oof, gbc_test_oof = gbc_oof.get_oof()
    print("gbc model OK !")
    # get train_oof and test_oof from rf model
    rf_oof = OutOfFold(clf=rf_ma, train=train, train_label=train_label, test=test)
    rf_oof.set_skf()
    rf_train_oof, rf_test_oof = rf_oof.get_oof()
    print("rf model OK !")
    # get train_oof and test_oof from ab model
    ab_oof = OutOfFold(clf=ab_ma, train=train, train_label=train_label, test=test)
    ab_oof.set_skf()
    ab_train_oof, ab_test_oof = ab_oof.get_oof()
    print("ab model OK !")
    # get train_oof and test_oof from lr model
    lr_oof = OutOfFold(clf=lr_ma, train=train, train_label=train_label, test=test)
    lr_oof.set_skf()
    lr_train_oof, lr_test_oof = lr_oof.get_oof()
    print("lr model OK !")
    # get train_oof and test_oof from et model
    et_oof = OutOfFold(clf=et_ma, train=train, train_label=train_label, test=test)
    et_oof.set_skf()
    et_train_oof, et_test_oof = et_oof.get_oof()
    print("et model OK !")
    # get train_oof and test_oof from kn model
    kn_oof = OutOfFold(clf=kn_ma, train=train, train_label=train_label, test=test)
    kn_oof.set_skf()
    kn_train_oof, kn_test_oof = kn_oof.get_oof()
    print("kn model OK !")

    ######
    mda = (ModelDataAgg(gbc_train_oof.shape[0], gbc_test_oof.shape[0],
                        gbc_train_oof=gbc_train_oof, rf_train_oof=rf_train_oof, ab_train_oof=ab_train_oof,
                        lr_train_oof=lr_train_oof, et_train_oof=et_train_oof, kn_train_oof=kn_train_oof,
                        gbc_test_oof=gbc_test_oof, rf_test_oof=rf_test_oof, ab_test_oof=ab_test_oof,
                        lr_test_oof=lr_test_oof, et_test_oof=et_test_oof, kn_test_oof=kn_test_oof))
    mda.train_test_split()
    meta_train = mda.train_merge()
    meta_test = mda.test_merge()
    print("train data OK !")
    print("test data OK !")
    print(meta_train.shape)
    print(meta_test.shape)

    ######
    xm = XgbModel(train=np.hstack((train, meta_train)),
                  train_label=train_label,
                  test=np.hstack((test, meta_test)), test_label=test_label)
    xm.train()
    xm.predict()
    xm.evaluate()
    xm.evaluate_output() 