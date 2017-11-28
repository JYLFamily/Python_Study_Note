# coding:utf-8

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


class ModelAssistant(object):

    def __init__(self, clf, random_state=9, params=None):
        self.__random_state = random_state
        self.__params = dict() if params is None else params
        if clf is KNeighborsClassifier:
            self.__clf = clf(** self.__params)
        elif clf is XGBClassifier:
            self.__params["seed"] = self.__random_state
            self.__clf = clf(** self.__params)
        else:
            self.__clf = clf(**self.__params)

    def train(self, train, train_label):
        self.__clf.fit(train, train_label)

    def predict(self, test):
        return self.__clf.predict_proba(test)[:, 1].reshape((-1, 1))