# coding:utf-8


class ModelAssistant(object):

    def __init__(self, clf, seed=9, params=None):
        self.__seed = seed
        self.__params = params
        self.__params["random_state"] = self.__seed
        self.__clf = clf(** self.__params)

    def train(self, train, train_label):
        self.__clf.fit(train, train_label)

    def predict(self, test):
        return self.__clf.predict_proba(test)[:, 1].reshape((test.shape[0], 1))