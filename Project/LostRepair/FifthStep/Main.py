# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Dense
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


class Main(object):

    def __init__(self, input_path, test_size, random_state, cv):
        self.__X = pd.read_csv(input_path, usecols=list(range(1, 53))).values
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

    def train_test_scale(self):
        self.__scaler = StandardScaler()
        self.__scaler.fit(self.__train)
        self.__train = self.__scaler.transform(self.__train)
        self.__test = self.__scaler.transform(self.__test)

    def stage_one(self, model_list):
        self.__skf = StratifiedKFold(n_splits=self.__cv, shuffle=True, random_state=self.__random_state)

        def get_oof_train(model):
            oof_train = np.zeros((self.__train.shape[0], 1))

            for i, (train_index, test_index) in enumerate(self.__skf.split(self.__train, self.__train_label)):
                x_train = self.__train[train_index]
                y_train = self.__train_label[train_index]
                x_test = self.__train[test_index]
                model.fit(x_train, y_train)
                # Sequential 对象注意下
                if isinstance(model, Sequential):
                    oof_train[test_index] = model.predict_proba(x_test).reshape((-1, 1))
                else:
                    oof_train[test_index] = model.predict_proba(x_test)[:, 1].reshape((-1, 1))

            return oof_train

        def get_oof_test(model):
            model.fit(self.__train, self.__train_label)
            # Sequential 对象注意下
            if isinstance(model, Sequential):
                oof_test = model.predict_proba(self.__test).reshape((-1, 1))
            else:
                oof_test = model.predict_proba(self.__test)[:, 1].reshape((-1, 1))

            return oof_test

        self.__oof_train_all = np.hstack(tuple(map(get_oof_train, model_list)))
        self.__oof_test_all = np.hstack(tuple(map(get_oof_test, model_list)))
        self.__train_all = np.hstack((self.__train, self.__oof_train_all))
        self.__test_all = np.hstack((self.__test, self.__oof_test_all))

    def stage_two(self, model):
        model.fit(self.__train_all, self.__train_label)
        print(roc_auc_score(self.__test_label, model.predict(self.__test_all)))


if __name__ == "__main__":
    m = Main(input_path="D:\\Project\\LostRepair\\yunyingshang\\all.csv", test_size=0.2, random_state=9, cv=5)
    m.train_test_split()

    # stage one
    # GBDT ---------------------------
    Gbdt = GradientBoostingClassifier()
    # RF -----------------------------
    Rf = RandomForestClassifier()
    # DNN ----------------------------
    Dnn = Sequential([
                     Dense(input_dim=52, units=600, activation="relu"),
                     Dense(units=600, activation="relu"),
                     Dense(units=600, activation="relu"),
                     Dense(units=600, activation="relu"),
                     Dense(units=1, activation="sigmoid")])
    Dnn.compile(loss="binary_crossentropy", optimizer="adam")
    # LR -----------------------------
    Lr = LogisticRegression()
    m.stage_one([Gbdt, Rf, Dnn, Lr])

    # stage two
    bst = XGBClassifier()
    m.stage_two(bst)