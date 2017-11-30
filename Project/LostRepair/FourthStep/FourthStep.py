# coding:utf-8

import os
import re
import numpy as np
from sklearn.externals import joblib


class FourthStep(object):

    def __init__(self, input_path):
        self.__input_path = input_path
        self.__ada_boost_classifier = None
        self.__gradient_boosting_classifier = None
        self.__random_forest_classifier = None
        self.__logistic_regression = None
        self.__k_neighbors_classifier = None
        self.__extra_tree_classifier = None
        self.__xgb_classifier = None
        self.__model_list = []
        self.__bst = None
        self.__files = None
        self.__raw_features = None
        self.__model_features_list = []
        self.__model_features_array = None
        self.__raw_model_features = None

    def set_estimators(self):
        self.__files = os.listdir(self.__input_path)

        for file in self.__files:
            if re.search(r"pkl", file):
                if re.search(r"adboost", file):
                    self.__ada_boost_classifier = joblib.load(os.path.join(self.__input_path, file))
                if re.search(r"bst", file):
                    self.__bst = joblib.load(os.path.join(self.__input_path, file))
                if re.search(r"ET", file):
                    self.__extra_tree_classifier = joblib.load(os.path.join(self.__input_path, file))
                if re.search(r"GBDT", file):
                    self.__gradient_boosting_classifier = joblib.load(os.path.join(self.__input_path, file))
                if re.search(r"KNN", file):
                    self.__k_neighbors_classifier = joblib.load(os.path.join(self.__input_path, file))
                if re.search(r"LR", file):
                    self.__logistic_regression = joblib.load(os.path.join(self.__input_path, file))
                if re.search(r"RF", file):
                    self.__random_forest_classifier = joblib.load(os.path.join(self.__input_path, file))
                if re.search(r"xgb", file):
                    self.__xgb_classifier = joblib.load(os.path.join(self.__input_path, file))

        self.__model_list.extend([self.__ada_boost_classifier,
                                  self.__gradient_boosting_classifier,
                                  self.__random_forest_classifier,
                                  self.__logistic_regression,
                                  self.__k_neighbors_classifier,
                                  self.__extra_tree_classifier,
                                  self.__xgb_classifier])

    def stage_one_predict(self, raw_features):
        if type(raw_features) != np.ndarray:
            raise TypeError("Need Numpy Array Object")
        else:
            self.__raw_features = raw_features.reshape((1, -1))

        for model in self.__model_list:
            self.__model_features_list.append(model.predict_proba(self.__raw_features)[:, 1])

        print(self.__model_features_list)
        self.__model_features_array = np.array(self.__model_features_list).reshape((1, -1))
        self.__raw_model_features = np.hstack((self.__raw_features, self.__model_features_array))

    def stage_two_predict(self):
        return self.__bst.predict_proba(self.__raw_model_features)[:, 1]


if __name__ == "__main__":
    fs = FourthStep("C:\\Users\\Dell\\Desktop")
    fs.set_estimators()
    fs.stage_one_predict(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 450]))
    print(fs.stage_two_predict())
