# coding:utf-8

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from Project.LostRepair.LostRepairV1.FeatureEngineering import FeatureEngineering
from Project.LostRepair.LostRepairV1.NnGenerateFeature import NnGenerateFeature


class Main(object):

    def __init__(self, *, input_path, sep, header, test_size, random_state, cv):
        logging.basicConfig(filename="my.log",
                            filemode="w",
                            format="[%(asctime)s]-[%(name)s]-[%(lineno)d]-[%(levelname)s]-[%(message)s]",
                            level=logging.DEBUG)
        self.__input_path = input_path
        self.__X = pd.read_csv(self.__input_path, sep=sep, header=header, usecols=list(range(1, 4)))
        self.__y = pd.read_csv(self.__input_path, sep=sep, header=header, usecols=[0])
        self.__test_size, self.__random_state = test_size, random_state
        self.__cv = cv
        self.__train, self.__train_label, self.__test, self.__test_label = None, None, None, None

        self.__train_linear, self.__test_linear = None, None
        self.__train_tree, self.__test_tree = None, None
        self.__train_net, self.__test_net = None, None

        self.__train_output_layer, self.__test_output_layer = None, None

        self.__skf = None
        self.__oof_train_tree, self.__oof_test_tree = None, None
        self.__oof_train_linear, self.__oof_test_linear = None, None

        self.__oof_train_all, self.__oof_test_all = None, None
        self.__train_all, self.__test_all = None, None

    def train_test_split(self):
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

        self.__train_label = self.__train_label.values
        self.__test_label = self.__test_label.values
        logging.info("train test split compelet.")

    def feature_engineering(self):
        self.__train_linear, self.__test_linear = (
            FeatureEngineering.linear_model_feature_engineering(train=self.__train,
                                                                test=self.__test)
        )
        logging.info("linear model feature engineering complete")

        self.__train_tree, self.__test_tree = (
            FeatureEngineering.tree_model_feature_engineering(train=self.__train,
                                                              test=self.__test)
        )
        logging.info("tree model feature engineering complete")

        self.__train_net, self.__test_net = (
            FeatureEngineering.net_model_feature_engineering(train=self.__train,
                                                             test=self.__test)
        )
        logging.info("net model feature engineering complete")

    def nn_generate_feature(self):
        self.__train_output_layer, self.__test_output_layer = (
            NnGenerateFeature.get_intermediate_layer_output(train=self.__train_net, train_label=self.__train_label,
                                                            test=self.__test_net, test_label=self.__test_label)
        )
        logging.info("nn generate feature complete")

    def stage_one(self, *, tree_model_list, linear_model_list):
        try:
            self.__skf = StratifiedKFold(n_splits=self.__cv, shuffle=True, random_state=self.__random_state)

            def get_oof_train(model, train, train_label):
                oof_train = np.zeros((train.shape[0], 1))

                for i, (train_index, test_index) in enumerate(self.__skf.split(train, train_label)):
                    x_train = train[train_index]
                    y_train = train_label[train_index]
                    x_test = train[test_index]
                    model.fit(x_train, y_train)

                    oof_train[test_index] = model.predict_proba(x_test)[:, 1].reshape((-1, 1))

                return oof_train

            def get_oof_test(model, train, train_label, test):
                model.fit(train, train_label)
                oof_test = model.predict_proba(test)[:, 1].reshape((-1, 1))

                return oof_test

            self.__oof_train_tree = (
                np.hstack(tuple(map(get_oof_train,
                                    tree_model_list,
                                    [self.__train_tree] * len(tree_model_list),
                                    [self.__train_label] * len(tree_model_list))))
            )
            self.__oof_test_tree = (
                np.hstack(tuple(map(get_oof_test, tree_model_list,
                                    [self.__train_tree] * len(tree_model_list),
                                    [self.__train_label] * len(tree_model_list),
                                    [self.__test_tree] * len(tree_model_list))))
            )
            self.__oof_train_linear = (
                np.hstack(tuple(map(get_oof_train,
                                    linear_model_list,
                                    [self.__train_linear] * len(linear_model_list),
                                    [self.__train_label] * len(linear_model_list))))
            )
            self.__oof_test_linear = (
                np.hstack(tuple(map(get_oof_test, linear_model_list,
                                    [self.__train_linear] * len(linear_model_list),
                                    [self.__train_label] * len(linear_model_list),
                                    [self.__test_linear] * len(linear_model_list))))
            )

            self.__train_all = np.hstack((self.__train_tree,
                                          self.__train_output_layer,
                                          self.__oof_train_tree,
                                          self.__oof_train_linear))
            self.__test_all = np.hstack((self.__test_tree,
                                         self.__test_output_layer,
                                         self.__oof_test_tree,
                                         self.__oof_test_linear))

            logging.info("stage one compelet.")
        except Exception as e:
            logging.exception(e)
            raise

    def stage_two(self, model):
        try:
            model.fit(self.__train_all, self.__train_label)
            print(roc_auc_score(self.__test_label, model.predict_proba(self.__test_all)[:, 1]))
            logging.info("stage two compelet.")
        except Exception as e:
            logging.exception(e)
            raise

if __name__ == "__main__":
    m = Main(input_path="C:\\Users\\Dell\\Desktop\\model.txt",
             sep="\t", header=None, test_size=0.2, random_state=9, cv=5)

    m.train_test_split()
    m.feature_engineering()
    m.nn_generate_feature()

    # tree model
    XGB_tree = XGBClassifier()
    RF = RandomForestClassifier()
    # linear model
    LR = LogisticRegression()
    XGB_linear = XGBClassifier()

    m.stage_one(tree_model_list=[XGB_tree, RF], linear_model_list=[LR, XGB_linear])
    m.stage_two(RFE(XGB_tree))
