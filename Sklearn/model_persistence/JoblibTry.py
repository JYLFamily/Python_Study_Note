# coding:utf-8

from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib


class JoblibTry(object):

    def __init__(self):
        self.__model = DecisionTreeClassifier()

    def model_dump(self):
        joblib.dump(self.__model, "C:\\Users\\Dell\\Desktop\\drt.pkl.z", True)

    def model_load(self):
        self.__model = joblib.load("C:\\Users\\Dell\\Desktop\\drt.pkl.z")

    def model_other(self):
        pass


if __name__ == "__main__":
    jbt = JoblibTry()
    jbt.model_dump()
    jbt.model_load()