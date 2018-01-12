# coding:utf-8

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


class Main(object):

    def __init__(self, input_path, test_size, random_state, cv):
        logging.basicConfig(filename="my.log",
                            filemode="w",
                            format="[%(asctime)s]-[%(name)s]-[%(lineno)d]-[%(levelname)s]-[%(message)s]",
                            level=logging.DEBUG)
        self.__X = pd.read_csv(input_path, usecols=list(range(1, 4))).values
        self.__y = pd.read_csv(input_path, usecols=[0]).values.ravel()
        self.__test_size, self.__random_state = test_size, random_state
        self.__cv = cv
        self.__train, self.__train_label, self.__test, self.__test_label = None, None, None, None
        self.__scaler = None
        self.__skf = None
        self.__oof_train_all, self.__oof_test_all = None, None
        self.__train_all, self.__test_all = None, None

    def train_test_split(self):
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))
        logging.info("train test split compelet.")

    def train_test_scale(self):
        self.__scaler = StandardScaler()
        self.__scaler.fit(self.__train)
        self.__train = self.__scaler.transform(self.__train)
        self.__test = self.__scaler.transform(self.__test)
        logging.info("train test scale compelet.")

    def stage_one(self, model_list):
        try:
            self.__skf = StratifiedKFold(n_splits=self.__cv, shuffle=True, random_state=self.__random_state)

            def get_oof_train(model):
                oof_train = np.zeros((self.__train.shape[0], model.n_estimators))

                for i, (train_index, test_index) in enumerate(self.__skf.split(self.__train, self.__train_label)):
                    x_train = self.__train[train_index]
                    y_train = self.__train_label[train_index]
                    x_test = self.__train[test_index]

                    model.fit(x_train, y_train)
                    oof_train[test_index] = model.apply(x_test)

                return oof_train

            def get_oof_test(model):
                model.fit(self.__train, self.__train_label)
                oof_test = model.apply(self.__test)

                return oof_test

            self.__oof_train_all = np.hstack(tuple(map(get_oof_train, model_list)))
            self.__oof_test_all = np.hstack(tuple(map(get_oof_test, model_list)))
            self.__train_all = np.hstack((self.__train, self.__oof_train_all))
            self.__test_all = np.hstack((self.__test, self.__oof_test_all))

            logging.info("stage one compelet.")
        except Exception as e:
            logging.exception(e)
            raise

    def stage_two(self, model):
        try:
            model.fit(self.__train_all, self.__train_label)
            print("Xgboost + RandomForest + LR %.4f" %
                  (roc_auc_score(self.__test_label, model.predict_proba(self.__test_all)[:, 1])))

            model.fit(self.__train, self.__train_label)
            print("LR %.4f" %
                  (roc_auc_score(self.__test_label, model.predict_proba(self.__test)[:, 1])))
            logging.info("stage two compelet.")
        except Exception as e:
            logging.exception(e)
            raise


if __name__ == "__main__":
    m = Main(input_path="C:\\Users\\Dell\\Desktop\\model2.csv", test_size=0.2, random_state=9, cv=5)
    m.train_test_split()
    m.train_test_scale()

    bst = XGBClassifier()
    rf = RandomForestClassifier()
    m.stage_one([bst, rf])

    lr = LogisticRegression()
    m.stage_two(lr)