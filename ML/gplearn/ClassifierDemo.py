# coding:utf-8

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class ClassifierDemo(object):

    def __init__(self):
        self.__digists = None
        self.__X = None
        self.__y = None
        self.__train, self.__test, self.__train_label, self.__test_label = [None for _ in range(4)]
        self.__train_gfeature, self.__test_gfeature = None, None

        self.__lr_baseline = None
        self.__lr_gfeature = None

    def data_prepare(self):
        self.__digists = load_digits(n_class=2)
        self.__X = self.__digists.data
        self.__y = self.__digists.target

        self.__train, self.__test, self.__train_label, self.__test_label = train_test_split(
            self.__X,
            self.__y,
            test_size=0.2,
            random_state=9
        )

        # standard scaler
        scaler = StandardScaler().fit(self.__train)
        self.__train = scaler.transform(self.__train)
        self.__test = scaler.transform(self.__test)

        # gp feature
        function_set = ("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min")

        gp = SymbolicTransformer(
            generations=5,
            population_size=2000,
            hall_of_fame=100,
            n_components=10,
            function_set=function_set,
            parsimony_coefficient=0.0005,
            max_samples=0.9,
            verbose=1,
            random_state=0,
            n_jobs=3
        )

        # 使用 stacking 的方式得到 generic feature 感觉更为合理
        gp.fit(self.__train, self.__train_label)
        self.__train_gfeature = np.hstack((self.__train, gp.transform(self.__train)))
        self.__test_gfeature = np.hstack((self.__test, gp.transform(self.__test)))
        # 能够得到生成衍生变量的公式
        # print(len(gp))

    def function_set(self):
        self.__lr_baseline = LogisticRegression()
        self.__lr_gfeature = LogisticRegression()

        self.__lr_baseline.fit(self.__train, self.__train_label)
        print("------------------------------------ base mode ------------------------------------")
        print(roc_auc_score(self.__test_label, self.__lr_baseline.predict_proba(self.__test)[:, 1]))

        self.__lr_gfeature.fit(self.__train_gfeature, self.__train_label)
        print("------------------------------------ gpfe mode ------------------------------------")
        print(roc_auc_score(self.__test_label, self.__lr_gfeature.predict_proba(self.__test_gfeature)[:, 1]))


if __name__ == "__main__":
    cd = ClassifierDemo()
    cd.data_prepare()
    cd.function_set()
