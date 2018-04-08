# coding:utf-8

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
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
        feature_selected = ["trans_num_max_by_date", "succ_trans_num_max_by_shop", "taobaoing_date", "open_last_days",
                            "tb0020003", "tb0020010", "ep0010004", "mp0010009", "ep0030009", "mp0010022",
                            "cnt_call_std", "tb0020004", "ep0060003", "ep0060004"]

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

        print("-" * 53)
        training_proba = self.__rf.predict_proba(feature)[:, 1]
        training_score = pd.Series(training_proba.reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x)))
        training_score_label = pd.DataFrame(np.hstack((training_score.reshape((-1, 1)), label.reshape((-1, 1)))))
        training_score_label.columns = ["score", "label"]
        training_score_label.to_csv("C:\\Users\\Dell\\Desktop\\training_score_label.csv", index=False)

        print("-" * 53)
        testing_proba = self.__rf.predict_proba(self.__test_feature)[:, 1]
        testing_score = pd.Series(testing_proba.reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x)))
        testing_score_label = pd.DataFrame(np.hstack((testing_score.reshape((-1, 1)), self.__test_label.reshape((-1, 1)))))
        testing_score_label.columns = ["score", "label"]
        testing_score_label.to_csv("C:\\Users\\Dell\\Desktop\\testing_score_label.csv", index=False)

    def model_persistence(self):
        rus = RandomUnderSampler(random_state=7)
        self.__test_feature, self.__test_label = rus.fit_sample(self.__test_feature, self.__test_label)
        feature = np.vstack((self.__train_feature, self.__validation_feature, self.__test_feature))
        label = np.vstack((self.__train_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)), self.__test_label.reshape((-1, 1)))).reshape((-1,))

        self.__rf.fit(feature, label)
        joblib.dump(self.__rf, "C:\\Users\\Dell\\Desktop\\rf.z")


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
    # rfb.model_persistence()
