# coding:utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import PredefinedSplit
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.utils import shuffle
from mlxtend.plotting import plot_learning_curves
from sklearn.externals import joblib


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

        self.__train_raw_feature = None
        self.__validation_raw_feature = None
        self.__test_raw_feature = None

        self.__train_raw_label = None
        self.__validation_raw_label = None
        self.__test_raw_label = None

        self.__categorical_header = categorical_header
        self.__numeric_header = None
        self.__train_categorical = None
        self.__train_numeric = None
        self.__validation_categorical = None
        self.__validation_numeric = None
        self.__test_categorical = None
        self.__test_numeric = None

        self.__rf = None
        self.__params = None
        self.__rf_bo = None
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

        self.__train_raw_label = self.__train_label.squeeze().values
        self.__validation_raw_label = self.__validation_label.squeeze().values
        self.__test_raw_label = self.__test_label.squeeze().values

        self.__train_label = self.__train_label.squeeze().values
        self.__validation_label = self.__validation_label.squeeze().values
        self.__test_label = self.__test_label.squeeze().values

    def pre_processing(self):
        # self.__numeric_header = [i for i in self.__train_feature.columns if i not in self.__categorical_header]
        #
        # self.__train_categorical = self.__train_feature[self.__categorical_header]
        # self.__train_numeric = self.__train_feature[self.__numeric_header]
        #
        # self.__validation_categorical = self.__validation_feature[self.__categorical_header]
        # self.__validation_numeric = self.__validation_feature[self.__numeric_header]
        #
        # self.__test_categorical = self.__test_feature[self.__categorical_header]
        # self.__test_numeric = self.__test_feature[self.__numeric_header]
        #
        # self.__train_categorical = self.__train_categorical.astype(str)
        # self.__validation_categorical = self.__validation_categorical.astype(str)
        # self.__test_categorical = self.__test_categorical.astype(str)
        #
        # self.__train_categorical = self.__train_categorical.fillna("missing")
        # self.__validation_categorical = self.__validation_categorical.fillna("missing")
        # self.__test_categorical = self.__test_categorical.fillna("missing")
        #
        # # 使用 DataFrameMapper 生成的 DataFrame 舍弃了之前 DataFrame 的 index 需要 set_index
        # mapper = DataFrameMapper([(i, LabelEncoder()) for i in self.__train_categorical.columns])
        # mapper.fit(self.__train_categorical)
        # self.__train_categorical = pd.DataFrame(mapper.transform(self.__train_categorical), columns=self.__train_categorical.columns).set_index(self.__train_categorical.index)
        # self.__validation_categorical = pd.DataFrame(mapper.transform(self.__validation_categorical), columns=self.__validation_categorical.columns).set_index(self.__validation_categorical.index)
        # self.__test_categorical = pd.DataFrame(mapper.transform(self.__test_categorical), columns=self.__test_categorical.columns).set_index(self.__test_categorical.index)

        feature_selected = ["trans_num_max_by_date", "succ_trans_num_max_by_shop", "taobaoing_date", "open_last_days",
                            "tb0020003", "tb0020010", "ep0010004", "mp0010009","ep0030009", "mp0010022", "cnt_call_std",
                            "tb0020004", "ep0060003", "ep0060004"]

        self.__train_numeric = self.__train_feature[feature_selected]
        self.__validation_numeric = self.__validation_feature[feature_selected]
        self.__test_numeric = self.__test_feature[feature_selected]

        self.__train_numeric = self.__train_numeric.fillna(-999)
        self.__validation_numeric = self.__validation_numeric.fillna(-999)
        self.__test_numeric = self.__test_numeric.fillna(-999)

        self.__train_raw_feature = self.__train_numeric.values
        self.__validation_raw_feature = self.__validation_numeric.values
        self.__test_raw_feature = self.__test_numeric.values

        self.__train_feature = self.__train_numeric.values
        self.__validation_feature = self.__validation_numeric.values
        self.__test_feature = self.__test_numeric.values

        # self.__train_feature = pd.concat([self.__train_categorical, self.__train_numeric], axis=1).values
        # self.__validation_feature = pd.concat([self.__validation_categorical, self.__validation_numeric], axis=1).values
        # self.__test_feature = pd.concat([self.__test_categorical, self.__test_numeric], axis=1).values

    def re_sample(self):
        rus = RandomUnderSampler(random_state=7)

        # train 进行重抽样
        self.__train_feature, self.__train_label = rus.fit_sample(self.__train_feature, self.__train_label)

        self.__train_validation_feature = np.vstack((self.__train_feature, self.__validation_feature))
        self.__train_validation_label = np.vstack((self.__train_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)))).reshape((-1, ))
        self.__train_validation_index = np.ones((self.__train_validation_label.shape[0], ))
        self.__train_validation_index[self.__train_label.shape[0]:] = -1

        # validation 进行重抽样
        rus = RandomUnderSampler(random_state=7)
        self.__validation_feature, self.__validation_label = rus.fit_sample(self.__validation_feature, self.__validation_label)

    def fit_predict(self):
        def __rf_cv(n_estimators, min_samples_split, max_features):
            ps = PredefinedSplit(self.__train_validation_index)
            clf = RandomForestClassifier(
                n_estimators=int(n_estimators),
                min_samples_split=int(min_samples_split),
                max_features=min(max_features, 0.999),
                random_state=7
            )
            val = cross_val_score(
                clf,
                self.__train_validation_feature,
                self.__train_validation_label,
                scoring="roc_auc",
                cv=ps
            ).mean()

            return val
        #
        # self.__params = {"n_estimators": (5, 250), "min_samples_split": (2, 25), "max_features": (0.1, 0.999)}
        # self.__rf_bo = BayesianOptimization(__rf_cv, self.__params, random_state=7)
        # self.__rf_bo.maximize(init_points=20, n_iter=100, kappa=10, ** {"alpha": 1e-5})

        # self.__rf = RandomForestClassifier(
        #     n_estimators=int(self.__rf_bo.res["max"]["max_params"]["n_estimators"]),
        #     min_samples_split=int(self.__rf_bo.res["max"]["max_params"]["min_samples_split"]),
        #     max_features=round(self.__rf_bo.res["max"]["max_params"]["max_features"], 4),
        #     random_state=7
        # )

        self.__rf = RandomForestClassifier(
            n_estimators=240,
            min_samples_split=28,
            max_features=0.1,
            random_state=7
        )

        feature = np.vstack((self.__train_feature, self.__validation_feature))
        label = np.vstack((self.__train_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)))).reshape((-1, ))
        # 类似 refit 的操作
        self.__rf.fit(
            feature,
            label
        )

        training_auc = round(roc_auc_score(label, self.__rf.predict(feature)), 4)
        testing_auc = round(roc_auc_score(self.__test_label, self.__rf.predict(self.__test_feature)), 4)

        # print("-" * 53)
        # print(self.__rf_bo.res["max"]["max_params"])
        print("-" * 53)
        print("training auc: " + str(training_auc))
        print("testing auc: " + str(testing_auc))

        training_recall = round(recall_score(label, self.__rf.predict(feature)), 4)
        testing_recall = round(recall_score(self.__test_label, self.__rf.predict(self.__test_feature)), 4)
        print("-" * 53)
        print("training recall: " + str(training_recall))
        print("testing recall: " + str(testing_recall))

    def model_persistence(self):
        rus = RandomUnderSampler(random_state=7)
        self.__test_feature, self.__test_label = rus.fit_sample(self.__test_feature, self.__test_label)
        feature = np.vstack((self.__train_feature, self.__validation_feature, self.__test_feature))
        label = np.vstack((self.__train_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)), self.__test_label.reshape((-1, 1)))).reshape((-1,))

        self.__rf.fit(feature, label)
        joblib.dump(self.__rf, "C:\\Users\\Dell\\Desktop\\rf.z")

    # def plot_learning_curves(self):
    #
    #     feature = np.vstack((self.__train_feature, self.__validation_feature))
    #     label = np.vstack((self.__train_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)))).reshape((-1, ))
    #
    #     feature, label = shuffle(feature, label, random_state=7)
    #     plot_learning_curves(feature, label, self.__test_feature, self.__test_label, self.__rf)
    #     plt.show()

    # def plot_confusion_matrix(self):
    #     self.__cm = confusion_matrix(y_target=self.__test_label, y_predicted=self.__test_predict)
    #     _, _ = plot_confusion_matrix(conf_mat=self.__cm)
    #     plt.show()
    #
    # def plot_feature_importance(self):
    #     temp = pd.DataFrame({"feature": self.__train_feature_columns, "importance": self.__rf.feature_importances_})
    #     temp = temp.sort_values(by="importance", ascending=False)
    #
    #     sns.barplot(x="importance", y="feature", data=temp)
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
    rfb.model_persistence()
    # rfb.plot_learning_curves()
