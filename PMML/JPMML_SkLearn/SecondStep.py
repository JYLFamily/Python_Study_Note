# coding:utf-8

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.externals import joblib


class SecondStep(object):

    def __init__(self, input_path, test_size, random_state, raw_features):
        self.__input_path = input_path
        self.__X = pd.read_csv(input_path, header=None, usecols=list(range(1, 32)))
        self.__y = pd.read_csv(input_path, header=None, usecols=[0])
        self.__test_size = test_size
        self.__random_state = random_state
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__gradient_boosting_classifier = GradientBoostingClassifier()
        self.__random_forest_classifier = RandomForestClassifier()
        self.__logistic_regression = LogisticRegression()
        self.__k_neighbors_classifier = KNeighborsClassifier()
        self.__extra_tree_classifier = ExtraTreeClassifier()
        self.__xgb_classifier = XGBClassifier()
        self.__model_list = []
        self.__pmml_model_list = []
        self.__pmml_model_name = []
        self.__files = None
        self.__raw_features = raw_features

    def train_test_split(self):
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

        self.__train = self.__train.values
        self.__train_label = self.__train_label.values.ravel()
        self.__test = self.__test.values
        self.__test_label = self.__test_label.values.ravel()

    def model_wrapper_fit(self):
        self.__model_list.extend([self.__gradient_boosting_classifier,
                                  self.__random_forest_classifier,
                                  self.__logistic_regression,
                                  self.__k_neighbors_classifier,
                                  self.__extra_tree_classifier,
                                  self.__xgb_classifier])

        for model in self.__model_list:
            temp = PMMLPipeline([("estimator", model)])
            temp.fit(self.__train, self.__train_label)
            self.__pmml_model_list.append(temp)

    def model_dump(self):
        self.__pmml_model_name.extend(["GBDTML", "RFML", "LRML", "KNNML", "ETML", "XGBML"])

        for pmml_model, model_name in zip(self.__pmml_model_list, self.__pmml_model_name):
            joblib.dump(pmml_model,  os.path.join(os.path.dirname(self.__input_path), model_name + ".pkl.z"),
                        compress=True)

    def model_load(self):
        self.__raw_features = list(json.loads(self.__raw_features).values())
        self.__raw_features = np.array(self.__raw_features).reshape((1, -1))

        self.__files = os.listdir(os.path.dirname(self.__input_path))
        for file in self.__files:
            if re.search(r"XGBML", file):
                self.__gradient_boosting_classifier = joblib.load(os.path.join(os.path.dirname(self.__input_path), file))
                print(self.__gradient_boosting_classifier.predict_proba(self.__raw_features)[:, 1])


if __name__ == "__main__":
    # ss = SecondStep("C:\\Users\\Dell\\Desktop\\features_all.csv", 0.2, 9)
    # ss.train_test_split()
    # ss.model_wrapper_fit()
    # ss.model_dump()

    features = ('{"f0":0, "f1":0, "f2":0, "f3":1, "f4":1,'
                ' "f5":1, "f6":0, "f7":0, "f8":0, "f9":0,'
                ' "f10":0, "f11":0, "f12":0, "f13":0, "f14":0,'
                ' "f15":0, "f16":0, "f17":0, "f18":0, "f19":0,'
                ' "f20":1, "f21":0, "f22":0, "f23":0, "f24":0,'
                ' "f25":0, "f26":0, "f27":0, "f28":1, "f29":0, "f30":300}')

    ss = SecondStep("C:\\Users\\Dell\\Desktop\\features_all.csv", 0.2, 9, features)
    ss.train_test_split()
    ss.model_wrapper_fit()
    ss.model_dump()
    ss.model_load()
