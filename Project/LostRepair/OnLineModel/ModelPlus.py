# coding:utf-8

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from sklearn.externals import joblib


class ModelPlus(object):

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
        self.__raw_df = None
        self.__index = None
        self.__raw_features = None
        self.__model_features = None
        self.__all_features = None
        self.__return_proba = None
        self.__features_name = ["f0",
                                "f1",
                                "f2",
                                "f3",
                                "f4",
                                "f5",
                                "f6",
                                "f7",
                                "f8",
                                "f9",
                                "f10",
                                "f11",
                                "f12",
                                "f13",
                                "f14",
                                "f15",
                                "f16",
                                "f17",
                                "f18",
                                "f19",
                                "f20",
                                "f21",
                                "f22",
                                "f23",
                                "f24",
                                "f25",
                                "f26",
                                "f27",
                                "f28",
                                "f29",
                                "f30"]

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

    def return_predict(self, raw):
        try:
            self.__raw_df = pd.DataFrame(json.loads(raw)).T
            # self.__raw_df = pd.read_json(raw, orient="index", dtype="object")
            # index 电话号码
            self.__index = list(self.__raw_df.index)
            # raw_features 电话号码对应原始特征
            self.__raw_features = self.__raw_df[self.__features_name].values.reshape((-1, len(self.__features_name)))

            def model_features_calculate(x):
                return x.predict_proba(self.__raw_features)[:, 1].reshape((-1, 1))

            self.__model_features = np.hstack(tuple(map(model_features_calculate, self.__model_list)))
            self.__all_features = np.hstack((self.__raw_features, self.__model_features))

            self.__return_proba = pd.Series(self.__bst.predict_proba(self.__all_features)[:, 1], index=self.__index)
            logging.info("return_proba complete")

            return self.__return_proba.to_json(orient="index")

        except Exception as e:
            logging.exception(e)


if __name__ == "__main__":
    # features = '{ \
    #     "1235":{"nameEquesSpecialChar":0,"close_order":450,"KG0113":69,"KG0110":0,"familyNameEquals":0,"KG0111":0,' \
    #         '"nameStartWithA":0,"KG0109A":1,"nameContainsKeyWords":0,"nameContainsRelative":0,"KG0105":0,"KG0106":0,' \
    #         '"KG0103":0,"KG0202":0,"KG0104":0,"KG0203":0,"KG0401":1,"KG0109":0,"KG0107":0,"KG0108":0,' \
    #         '"nameContainsWorkRelation":0,"KG0101":0,"KG0102":0,"nameSameChar":0,"lengthGreaterThree":1,"KG0101A":0,' \
    #         '"nameContainsLikeRelative":0,"lengthEquealOne":0,"KG0114":0,"KG0115":0,"close_score":0},\
    #     "1234":{"nameEquesSpecialChar":0,"close_order":450,"KG0113":69,"KG0110":0,"familyNameEquals":0,"KG0111":0,' \
    #         '"nameStartWithA":0,"KG0109A":1,"nameContainsKeyWords":0,"nameContainsRelative":0,"KG0105":0,"KG0106":0,' \
    #         '"KG0103":0,"KG0202":0,"KG0104":0,"KG0203":0,"KG0401":1,"KG0109":0,"KG0107":0,"KG0108":0,' \
    #         '"nameContainsWorkRelation":0,"KG0101":0,"KG0102":0,"nameSameChar":0,"lengthGreaterThree":1,"KG0101A":0,' \
    #         '"nameContainsLikeRelative":0,"lengthEquealOne":0,"KG0114":0,"KG0115":0,"close_score":0} \
    #     }'

    features = '{ \
           "01235":{"f6":0,"f30":450,"f23":69,"f21":0,"f7":0,"f22":0,' \
               '"f2":0,"f20":1,"f5":0,"f3":0,"f15":0,"f16":0,' \
               '"f13":0,"f26":0,"f14":0,"f27":0,"f28":1,"f19":0,"f17":0,"f18":0,' \
               '"f8":0,"f10":0,"f12":0,"f9":0,"f1":1,"f11":0,' \
               '"f4":0,"f0":0,"f24":0,"f25":0,"f29":0},\
           "01234":{"f6":0,"f30":450,"f23":69,"f21":0,"f7":0,"f22":0,' \
               '"f2":0,"f20":1,"f5":0,"f3":0,"f15":0,"f16":0,' \
               '"f13":0,"f26":0,"f14":0,"f27":0,"f28":1,"f19":0,"f17":0,"f18":0,' \
               '"f8":0,"f10":0,"f12":0,"f9":0,"f1":1,"f11":0,' \
               '"f4":0,"f0":0,"f24":0,"f25":0,"f29":0} \
           }'

    mp = ModelPlus("C:\\Users\\Dell\\Desktop\\week")
    mp.set_estimators()
    print(mp.return_predict(features))