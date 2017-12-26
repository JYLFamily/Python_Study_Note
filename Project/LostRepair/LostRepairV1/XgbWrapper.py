# coding:utf-8


import xgboost as xgb


class XgbWrapper(object):

    def __init__(self, *, booster):
        self.__params = {"booster": booster}
        self.__bst = None

    def fit(self, train, train_label):
        self.__bst = xgb.train(self.__params, xgb.DMatrix(data=train[0:100], label=train_label[0:100]))

    def predict_proba(self, test):
        return self.__bst.predict(xgb.DMatrix(data=test))


if __name__ == "__main__":
    pass