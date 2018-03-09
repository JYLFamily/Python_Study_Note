# coding:utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


class RandomForestBaseline(object):

    def __init__(self, *, input_path, categorical_header):
        self.__input_path = input_path
        self.__train_feature = None
        self.__train_label = None
        self.__test_feature = None
        self.__test_label = None
        self.__train_feature_columns = None
        self.__train_label_positive_num = None
        self.__train_label_negative_num = None

        self.__categorical_header = categorical_header
        self.__numeric_header = None
        self.__train_categorical = None
        self.__train_numeric = None
        self.__test_categorical = None
        self.__test_numeric = None

        self.__rf = None
        self.__test_predict = None

        self.__cm = None

    def read(self):
        self.__train_feature = pd.read_csv(os.path.join(self.__input_path, "train_feature.csv"), encoding="gbk")
        self.__train_feature = self.__train_feature.drop(
            ["user_label", "loan_status", "cb0180003"] + self.__categorical_header,
            axis=1
        )
        self.__train_label = pd.read_csv(os.path.join(self.__input_path, "train_label.csv")).squeeze().values

        self.__test_feature = pd.read_csv(os.path.join(self.__input_path, "test_feature.csv"), encoding="gbk")
        self.__test_feature = self.__test_feature.drop(
            ["user_label", "loan_status", "cb0180003"] + self.__categorical_header,
            axis=1
        )
        self.__test_label = pd.read_csv(os.path.join(self.__input_path, "test_label.csv")).squeeze().values
        self.__train_feature_columns = self.__train_feature.columns.values

    def set_validation(self):
        pass

    def pre_processing(self):
        self.__train_feature = self.__train_feature.fillna(-999).values
        self.__test_feature = self.__test_feature.fillna(-999).values

    def re_sample(self):
        rus = RandomUnderSampler()
        self.__train_feature, self.__train_label = rus.fit_sample(self.__train_feature, self.__train_label)

    def fit_predict(self):
        self.__rf = RandomForestClassifier()
        self.__rf.fit(self.__train_feature, self.__train_label)
        self.__test_predict = self.__rf.predict(self.__test_feature)

        training_auc = round(roc_auc_score(self.__train_label, self.__rf.predict_proba(self.__train_feature)[:, 1]), 4)
        testing_auc = round(roc_auc_score(self.__test_label, self.__rf.predict_proba(self.__test_feature)[:, 1]), 4)

        training_precision = round(precision_score(self.__train_label, self.__rf.predict(self.__train_feature)), 4)
        testing_precision = round(precision_score(self.__test_label, self.__rf.predict(self.__test_feature)), 4)

        training_recall = round(recall_score(self.__train_label, self.__rf.predict(self.__train_feature)), 4)
        testing_recall = round(recall_score(self.__test_label, self.__rf.predict(self.__test_feature)), 4)

        print("training auc: " + str(training_auc))
        print("testing auc: " + str(testing_auc))
        print("------------------------------------------------")
        print("training precision: " + str(training_precision))
        print("testing precision: " + str(testing_precision))
        print("------------------------------------------------")
        print("training recall: " + str(training_recall))
        print("testing recall: " + str(testing_recall))

    def plot_feature_importance(self):
        df = pd.DataFrame({"feature": self.__train_feature_columns, "importance": self.__rf.feature_importances_})
        df = df.sort_values(by="importance", ascending=False)
        sns.barplot(
            x="importance",
            y="feature",
            data=df
        )
        plt.show()

    def plot_confusion_matrix(self):
        self.__cm = confusion_matrix(y_target=self.__test_label, y_predicted=self.__test_predict)
        _, _ = plot_confusion_matrix(conf_mat=self.__cm)
        plt.show()


if __name__ == "__main__":
    rfb = RandomForestBaseline(
        input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data",
        categorical_header=["cb0180003", "ep0030004", "province_name", "cb0180003", "ep0030004", "province_name"],
    )
    rfb.read()
    rfb.pre_processing()
    rfb.re_sample()
    rfb.fit_predict()
    rfb.plot_confusion_matrix()


