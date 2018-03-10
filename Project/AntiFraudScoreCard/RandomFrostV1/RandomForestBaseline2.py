# coding:utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
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
        self.__train_feature_columns = None
        self.__test_feature = None
        self.__test_label = None

        self.__validation_feature = None
        self.__validation_label = None

        self.__train_validation_feature = None
        self.__train_validation_label = None
        self.__train_validation_index = None

        self.__categorical_header = categorical_header
        self.__numeric_header = None
        self.__train_categorical = None
        self.__train_numeric = None
        self.__validation_categorical = None
        self.__validation_numeric = None
        self.__test_categorical = None
        self.__test_numeric = None

        self.__rf = None
        self.__ps = None
        self.__params = None
        self.__clf = None
        self.__test_predict = None

        self.__cm = None

    def read(self):
        self.__train_feature = pd.read_csv(os.path.join(self.__input_path, "train_feature.csv"), encoding="gbk")
        self.__train_feature = self.__train_feature.drop(["user_label", "loan_status", "cb0180003"], axis=1)
        self.__train_feature_columns = self.__train_feature.columns.values
        self.__train_label = pd.read_csv(os.path.join(self.__input_path, "train_label.csv"))

        self.__test_feature = pd.read_csv(os.path.join(self.__input_path, "test_feature.csv"), encoding="gbk")
        self.__test_feature = self.__test_feature.drop(["user_label", "loan_status", "cb0180003"], axis=1)
        self.__test_label = pd.read_csv(os.path.join(self.__input_path, "test_label.csv"))

    def set_validation(self):
        self.__validation_feature, self.__test_feature, self.__validation_label, self.__test_label = (
            train_test_split(self.__test_feature, self.__test_label, test_size=0.8, random_state=7, shuffle=False)
        )
        self.__train_label = self.__train_label.squeeze().values
        self.__validation_label = self.__validation_label.squeeze().values
        self.__test_label = self.__test_label.squeeze().values

    def pre_processing(self):
        self.__numeric_header = [i for i in self.__train_feature.columns if i not in self.__categorical_header]

        self.__train_categorical = self.__train_feature[self.__categorical_header]
        self.__train_numeric = self.__train_feature[self.__numeric_header]

        self.__validation_categorical = self.__validation_feature[self.__categorical_header]
        self.__validation_numeric = self.__validation_feature[self.__numeric_header]

        self.__test_categorical = self.__test_feature[self.__categorical_header]
        self.__test_numeric = self.__test_feature[self.__numeric_header]

        self.__train_categorical = self.__train_categorical.astype(str)
        self.__validation_categorical = self.__validation_categorical.astype(str)
        self.__test_categorical = self.__test_categorical.astype(str)

        self.__train_categorical = self.__train_categorical.fillna("missing")
        self.__validation_categorical = self.__validation_categorical.fillna("missing")
        self.__test_categorical = self.__test_categorical.fillna("missing")

        # 使用 DataFrameMapper 生成的 DataFrame 舍弃了之前 DataFrame 的 index 需要 set_index
        mapper = DataFrameMapper([(i, LabelEncoder()) for i in self.__train_categorical.columns])
        mapper.fit(self.__train_categorical)
        self.__train_categorical = pd.DataFrame(mapper.transform(self.__train_categorical), columns=self.__train_categorical.columns).set_index(self.__train_categorical.index)
        self.__validation_categorical = pd.DataFrame(mapper.transform(self.__validation_categorical), columns=self.__validation_categorical.columns).set_index(self.__validation_categorical.index)
        self.__test_categorical = pd.DataFrame(mapper.transform(self.__test_categorical), columns=self.__test_categorical.columns).set_index(self.__test_categorical.index)

        self.__train_numeric = self.__train_numeric.fillna(-999)
        self.__validation_numeric = self.__validation_numeric.fillna(-999)
        self.__test_numeric = self.__test_numeric.fillna(-999)

        self.__train_feature = pd.concat([self.__train_categorical, self.__train_numeric], axis=1).values
        self.__validation_feature = pd.concat([self.__validation_categorical, self.__validation_numeric], axis=1).values
        self.__test_feature = pd.concat([self.__test_categorical, self.__test_numeric], axis=1).values

    def re_sample(self):
        print(Counter(self.__train_label))
        cc = ClusterCentroids(random_state=7, n_jobs=-1)
        self.__train_feature, self.__train_label = cc.fit_sample(self.__train_feature, self.__train_label)

        self.__train_validation_feature = np.vstack((self.__train_feature, self.__validation_feature))
        self.__train_validation_label = np.vstack((self.__train_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)))).reshape((-1, ))
        self.__train_validation_index = np.ones((self.__train_validation_label.shape[0], ))
        self.__train_validation_index[self.__train_label.shape[0]:] = -1

    def fit_predict(self):
        self.__rf = RandomForestClassifier(random_state=7)
        self.__ps = PredefinedSplit(self.__train_validation_index)
        self.__params = {
            "n_estimators": list(range(55, 100, 5)),
            "min_samples_leaf": [25, 30, 35, 40, 45]
        }
        self.__clf = GridSearchCV(
            estimator=self.__rf,
            param_grid=self.__params,
            scoring="roc_auc",
            n_jobs=-1,
            refit=False,
            cv=self.__ps
        )
        self.__clf.fit(self.__train_validation_feature, self.__train_validation_label)
        self.__rf = RandomForestClassifier(** self.__clf.best_params_, random_state=7)
        self.__rf.fit(self.__train_feature, self.__train_label)
        training_recall = round(roc_auc_score(self.__train_label, self.__rf.predict(self.__train_feature)), 4)
        validation_recall = round(roc_auc_score(self.__validation_label, self.__rf.predict(self.__validation_feature)), 4)
        testing_recall = round(roc_auc_score(self.__test_label, self.__rf.predict(self.__test_feature)), 4)

        print("training roc_auc: " + str(training_recall))
        print("validation roc_auc: " + str(validation_recall))
        print("testing roc_auc: " + str(testing_recall))

        # training_auc = round(roc_auc_score(self.__train_label, self.__rf.predict_proba(self.__train_feature)[:, 1]), 4)
        # testing_auc = round(roc_auc_score(self.__test_label, self.__rf.predict_proba(self.__test_feature)[:, 1]), 4)
        #
        # training_precision = round(precision_score(self.__train_label, self.__rf.predict(self.__train_feature)), 4)
        # testing_precision = round(precision_score(self.__test_label, self.__rf.predict(self.__test_feature)), 4)
        #
        # training_recall = round(recall_score(self.__train_label, self.__rf.predict(self.__train_feature)), 4)
        # testing_recall = round(recall_score(self.__test_label, self.__rf.predict(self.__test_feature)), 4)
        #
        # print("training auc: " + str(training_auc))
        # print("testing auc: " + str(testing_auc))
        # print("------------------------------------------------")
        # print("training precision: " + str(training_precision))
        # print("testing precision: " + str(testing_precision))
        # print("------------------------------------------------")
        # print("training recall: " + str(training_recall))
        # print("testing recall: " + str(testing_recall))

    def plot_feature_importance(self):
        temp = pd.DataFrame({"feature": self.__train_feature_columns, "importance": self.__rf.feature_importances_})
        temp = temp.sort_values(by="importance", ascending=False)

        sns.barplot(x="importance", y="feature", data=temp)
        plt.show()

    # def plot_confusion_matrix(self):
    #     self.__cm = confusion_matrix(y_target=self.__test_label, y_predicted=self.__test_predict)
    #     _, _ = plot_confusion_matrix(conf_mat=self.__cm)
    #     plt.show()


if __name__ == "__main__":
    rfb = RandomForestBaseline(
        input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data",
        categorical_header=["ep0030004", "province_name"],
    )
    rfb.read()
    rfb.set_validation()
    rfb.pre_processing()
    rfb.re_sample()
    rfb.fit_predict()
    rfb.plot_feature_importance()


