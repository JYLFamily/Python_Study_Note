# coding:utf-8

import os
import re
import json
import logging
import numpy as np
from sklearn.externals import joblib


class Model(object):

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
        self.__raw_features_name = ["lengthEquealOne",
                                    "lengthGreaterThree",
                                    "nameStartWithA",
                                    "nameContainsRelative",
                                    "nameContainsLikeRelative",
                                    "nameContainsKeyWords",
                                    "nameEquesSpecialChar",
                                    "familyNameEquals",
                                    "nameContainsWorkRelation",
                                    "nameSameChar",
                                    "KG0101",
                                    "KG0101A",
                                    "KG0102",
                                    "KG0103",
                                    "KG0104",
                                    "KG0105",
                                    "KG0106",
                                    "KG0107",
                                    "KG0108",
                                    "KG0109",
                                    "KG0109A",
                                    "KG0110",
                                    "KG0111",
                                    "KG0113",
                                    "KG0114",
                                    "KG0115",
                                    "KG0202",
                                    "KG0203",
                                    "KG0401",
                                    "close_score",
                                    "close_order"]
        # key phone_number
        # value np.array features
        self.__raw_features_array = dict()
        self.__model_features_array = dict()
        self.__return_proba = dict()

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

        self.__model_list.extend([self.__logistic_regression,
                                  self.__k_neighbors_classifier,
                                  self.__extra_tree_classifier,
                                  self.__ada_boost_classifier,
                                  self.__gradient_boosting_classifier,
                                  self.__xgb_classifier,
                                  self.__random_forest_classifier])
        logging.info("estimators set complete")

    def return_predict(self, raw_features):
        try:
            self.__raw_features = json.loads(raw_features, encoding="bytes")

            raw_feature_temp = []
            for phone_number, features_tuple in self.__raw_features.items():
                for feature_name in self.__raw_features_name:
                    raw_feature_temp.append(features_tuple[feature_name])
                self.__raw_features_array[phone_number] = np.array(raw_feature_temp).reshape((1, -1))
                raw_feature_temp = []
            model_feature_temp = []
            logging.info("raw_features_array complete")

            for phone_number, features_array in self.__raw_features_array.items():
                for model in self.__model_list:
                    model_feature_temp.append(model.predict_proba(features_array)[:, 1][0])
                self.__model_features_array[phone_number] = np.hstack((features_array,
                                                                       np.array(model_feature_temp).reshape((1, -1))))
                model_feature_temp = []
            logging.info("model_features_array complete")

            for phone_number, features_array in self.__model_features_array.items():
                # numpy.float32 与 np.float64 is not JSON serializable
                self.__return_proba[phone_number] = float(self.__bst.predict_proba(features_array)[:, 1][0])

            self.__return_proba = json.dumps(self.__return_proba)
            logging.info("return_proba complete")
        except Exception as e:
            logging.exception(e)

        return self.__return_proba