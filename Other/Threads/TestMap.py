# coding:utf-8

import time
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from multiprocessing.dummy import Pool as ThreadPool


class TestMap(object):

    def __init__(self):
        self.__X = load_iris().data[0:100]
        self.__y = load_iris().target[0:100]
        self.__lr = LogisticRegression()
        self.__rf = RandomForestClassifier()
        self.__gbdt = GradientBoostingClassifier()
        self.__dt = DecisionTreeClassifier()
        self.__knn = KNeighborsClassifier()
        self.__model_list = [self.__lr, self.__rf, self.__gbdt, self.__dt, self.__knn]

    def for_loop(self):
        tic = time.time()
        for model in self.__model_list:
            model.fit(self.__X, self.__y)
        toc = time.time()
        print(toc-tic)

    def map_only(self):
        tic = time.time()
        tuple(map(lambda model: model.fit(self.__X, self.__y), self.__model_list))
        toc = time.time()
        print(toc-tic)

    def map_multiprocessing(self):
        pool = ThreadPool(4)
        tic = time.time()
        tuple(pool.map(lambda model: model.fit(self.__X, self.__y), self.__model_list))
        toc = time.time()
        print(toc-tic)
        pool.close()
        pool.join()


if __name__ == "__main__":
    tm = TestMap()
    tm.for_loop()
    tm.map_only()
    tm.map_multiprocessing()
