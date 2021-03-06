# coding:utf-8

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score


class RandomForestBaseline(object):

    def __init__(self, *, input_path, categorical_header):
        self.__input_path = input_path
        self.__train_feature = None
        self.__train_label = None
        self.__test_feature = None
        self.__test_label = None
        self.__categorical_header = categorical_header
        self.__numeric_header = None
        self.__train_categorical = None
        self.__train_numeric = None
        self.__test_categorical = None
        self.__test_numeric = None
        self.__rf = None

    def read(self):
        self.__train_feature = pd.read_csv(os.path.join(self.__input_path, "train_feature.csv"), encoding="gbk")
        self.__train_feature = self.__train_feature.drop(["user_label", "loan_status", "cb0180003"], axis=1)
        self.__train_label = pd.read_csv(os.path.join(self.__input_path, "train_label.csv")).squeeze().values

        self.__test_feature = pd.read_csv(os.path.join(self.__input_path, "test_feature.csv"), encoding="gbk")
        self.__test_feature = self.__test_feature.drop(["user_label", "loan_status", "cb0180003"], axis=1)
        self.__test_label = pd.read_csv(os.path.join(self.__input_path, "test_label.csv")).squeeze().values

        print(Counter(list(self.__train_label)))
        print(Counter(list(self.__test_label)))

    def pre_processing(self):
        self.__numeric_header = [i for i in self.__train_feature.columns if i not in self.__categorical_header]
        self.__train_categorical = self.__train_feature[self.__categorical_header]
        self.__train_numeric = self.__train_feature[self.__numeric_header]
        self.__test_categorical = self.__test_feature[self.__categorical_header]
        self.__test_numeric = self.__test_feature[self.__numeric_header]

        self.__train_categorical = self.__train_categorical.astype(str)
        self.__test_categorical = self.__test_categorical.astype(str)
        self.__train_categorical = self.__train_categorical.fillna("missing")
        self.__test_categorical = self.__test_categorical.fillna("missing")
        mapper = DataFrameMapper([(i, LabelEncoder()) for i in self.__train_categorical.columns])
        mapper.fit(self.__train_categorical)
        self.__train_categorical = pd.DataFrame(mapper.transform(self.__train_categorical), columns=self.__train_categorical.columns)
        self.__test_categorical = pd.DataFrame(mapper.transform(self.__test_categorical), columns=self.__test_categorical.columns)

        self.__train_numeric = self.__train_numeric.fillna(-999)
        self.__test_numeric = self.__test_numeric.fillna(-999)

        self.__train_feature = pd.concat([self.__train_numeric, self.__train_categorical], axis=1)
        self.__test_feature = pd.concat([self.__test_numeric, self.__test_categorical], axis=1)
        self.__train_feature = self.__train_feature.values
        self.__test_feature = self.__test_feature.values

    def fit_predict(self):
        self.__rf = RandomForestClassifier(random_state=7)
        self.__rf.fit(self.__train_feature, self.__train_label)
        training_auc = round(roc_auc_score(self.__train_label, self.__rf.predict_proba(self.__train_feature)[:, 1]), 4)
        testing_auc = round(roc_auc_score(self.__test_label, self.__rf.predict_proba(self.__test_feature)[:, 1]), 4)
        print("training auc: " + str(training_auc))
        print("testing auc: " + str(testing_auc))


if __name__ == "__main__":
    rfb = RandomForestBaseline(
        input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data",
        categorical_header=["ep0030004", "province_name"],
    )
    rfb.read()
    rfb.pre_processing()
    rfb.fit_predict()


