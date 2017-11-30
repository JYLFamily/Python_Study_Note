# coding:utf-8

import os
import re
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
from sklearn.externals import joblib


class StageOneOutput(object):

    def __init__(self, input_path, sep, test_size, random_state):
        self.__input_path = input_path
        self.__sep = sep
        self.__test_size = test_size
        self.__random_state = random_state
        self.__ada_boost_classifier = AdaBoostClassifier()
        self.__gradient_boosting_classifier = GradientBoostingClassifier()
        self.__random_forest_classifier = RandomForestClassifier()
        self.__logistic_regression = LogisticRegression()
        self.__k_neighbors_classifier = KNeighborsClassifier()
        self.__extra_tree_classifier = ExtraTreeClassifier()
        self.__xgb_classifier = XGBClassifier()
        self.__X = None
        self.__y = None
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__model_list = []
        self.__model_name_list = []
        self.__files = None

    def train_test_split(self):
        self.__X = pd.read_csv(self.__input_path, sep=self.__sep, header=None, usecols=list(range(1, 32)))
        self.__y = pd.read_csv(self.__input_path, sep=self.__sep, header=None, usecols=[0])
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

        self.__train = self.__train.values
        self.__test = self.__test.values
        self.__train_label = self.__train_label.values.ravel()
        self.__test_label = self.__test_label.values.ravel()

    def loop_model_output(self):
        self.__model_list.extend([self.__ada_boost_classifier,
                                  self.__gradient_boosting_classifier,
                                  self.__random_forest_classifier,
                                  self.__logistic_regression,
                                  self.__k_neighbors_classifier,
                                  self.__extra_tree_classifier,
                                  self.__xgb_classifier])
        self.__model_name_list.extend(["adboost", "GBDT", "RF", "LR", "KNN", "ET", "xgb"])

        for index, model in enumerate(self.__model_list):
            model.fit(self.__train, self.__train_label)
            joblib.dump(model, "C:\\Users\\Dell\\Desktop\\" + str(self.__model_name_list[index]) + ".pkl.z", compress=3)
            print("------ " + str(self.__model_name_list[index]) + " ------")

    def loop_model_input(self):
        self.__files = os.listdir("C:\\Users\\Dell\\Desktop")

        for file in self.__files:
            if re.search(r"pkl", file):
                if re.search(r"adboost", file):
                    self.__ada_boost_classifier = joblib.load(os.path.join("C:\\Users\\Dell\\Desktop", file))
                if re.search(r"ET", file):
                    self.__extra_tree_classifier = joblib.load(os.path.join("C:\\Users\\Dell\\Desktop", file))
                if re.search(r"GBDT", file):
                    self.__gradient_boosting_classifier = joblib.load(os.path.join("C:\\Users\\Dell\\Desktop", file))
                if re.search(r"KNN", file):
                    self.__k_neighbors_classifier = joblib.load(os.path.join("C:\\Users\\Dell\\Desktop", file))
                if re.search(r"LR", file):
                    self.__logistic_regression = joblib.load(os.path.join("C:\\Users\\Dell\\Desktop", file))
                if re.search(r"RF", file):
                    self.__random_forest_classifier = joblib.load(os.path.join("C:\\Users\\Dell\\Desktop", file))
                if re.search(r"xgb", file):
                    self.__xgb_classifier = joblib.load(os.path.join("C:\\Users\\Dell\\Desktop", file))

        print(self.__test[0:1])
        print(self.__gradient_boosting_classifier.predict_proba(self.__test[0:1]))


if __name__ == "__main__":
    soo = StageOneOutput("C:\\Users\\Dell\\Desktop\\features_all.csv", ",", 0.2, 9)
    soo.train_test_split()
    soo.loop_model_output()
    soo.loop_model_input()