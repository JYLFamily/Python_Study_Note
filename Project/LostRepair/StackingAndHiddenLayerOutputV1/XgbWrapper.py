# coding:utf-8

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class XgbWrapper(object):

    def __init__(self, *, booster="gblinear", objective="binary:logistic"):
        self.__params = {"booster": booster, "objective":objective}
        self.__bst = None

    def fit(self, train, train_label):
        self.__bst = xgb.train(self.__params, xgb.DMatrix(data=train, label=train_label))

    def predict_proba(self, test):
        return np.hstack((
            1 - self.__bst.predict(xgb.DMatrix(data=test)).reshape((-1, 1)),
            self.__bst.predict(xgb.DMatrix(data=test)).reshape((-1, 1))
        ))


if __name__ == "__main__":
    # data prepare
    iris = load_iris()
    X = iris.data[0:100]
    y = iris.target[0:100]
    train, test, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=9)

    # XgbWrapper
    xgb_linear = XgbWrapper()
    xgb_linear.fit(train, train_label)
    print(xgb_linear.predict_proba(test))