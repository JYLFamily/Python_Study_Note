# coding:utf-8

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score


class FirstStep(object):

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
        self.__bst = None
        self.__test_preds = None
        self.__test_predictions = None

    def set_train_test(self):
        self.__X = pd.read_csv(self.__input_path, header=None, sep=self.__sep, usecols=list(range(1, 4)))
        self.__y = pd.read_csv(self.__input_path, header=None, sep=self.__sep, usecols=[0])
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

    def fit(self):
        self.__bst = XGBClassifier(objective="binary:logistic")
        self.__bst.fit(self.__train
                       , self.__train_label
                       , eval_metric="auc")

    def predict(self):
        self.__test_preds = self.__bst.predict(self.__test)
        self.__test_predictions = [round(value) for value in self.__test_preds]

    def evaluate(self):
        fpr, tpr, thresholds = roc_curve(self.__test_label, self.__test_preds, pos_label=1)
        print("auc : " + str(auc(fpr, tpr)))
        print("accuracy score : " + str(accuracy_score(self.__test_label, self.__test_preds)))


if __name__ == "__main__":
    fs = FirstStep(input_path="C:\\Users\\Dell\\Desktop\\model2.csv")
    fs.set_train_test()
    fs.fit()
    fs.predict()
    fs.evaluate()