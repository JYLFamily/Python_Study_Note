# coding:utf-8

import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn2pmml import PMMLPipeline


class FirstStep(object):

    def __init__(self):
        self.__iris = load_iris()
        self.__X = pd.DataFrame(self.__iris.data, columns=self.__iris.feature_names)
        self.__y = pd.DataFrame(self.__iris.target, columns=["Species"])
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_one_sample = None
        self.__test_label = None
        self.__mapper = None
        self.__estimator = None
        self.__pipeline = None

    def train_test_split_step(self):
        self.__train, self.__test, self.__train_label, self.__test_label = (train_test_split(self.__X,
                                                                                             self.__y, test_size=0.2))
        self.__train = self.__train.reset_index(drop=True)
        self.__train_label = self.__train_label.reset_index(drop=True)
        self.__test = self.__test.reset_index(drop=True)
        self.__test_label = self.__train.reset_index(drop=True)

    def feature_engineering_step(self):
        self.__mapper = (DataFrameMapper([(["sepal length (cm)", "sepal width (cm)",
                                            "petal length (cm)", "petal width (cm)"],
                                           [StandardScaler()])]))

    def model_train_step(self):
        self.__estimator = DecisionTreeClassifier()

    def pipeline_step(self):
        self.__pipeline = PMMLPipeline([
            ("mapper", self.__mapper),
            ("estimator", self.__estimator)])
        self.__pipeline.fit(self.__train, self.__train_label)

    def output_step(self):
        joblib.dump(self.__pipeline, "C:\\Users\\Dell\\Desktop\\pipeline.pkl.z", compress=3)

    def input_step(self):
        self.__pipeline = joblib.load("C:\\Users\\Dell\\Desktop\\pipeline.pkl.z")
        self.__test_one_sample = self.__test[0:1]
        print(self.__pipeline.predict(self.__test))
        # 传入一行记录
        print(self.__pipeline.predict(self.__test_one_sample))


if __name__ == "__main__":
    fs = FirstStep()
    fs.train_test_split_step()
    fs.feature_engineering_step()
    fs.model_train_step()
    fs.pipeline_step()
    fs.output_step()
    fs.input_step()