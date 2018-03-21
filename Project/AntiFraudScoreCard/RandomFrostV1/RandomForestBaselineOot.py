# coding:utf-8

import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from collections import Counter


class RandomForestBaselineOot(object):

    def __init__(self, file_input_path, model_input_path):
        self.__loan = pd.read_csv(os.path.join(file_input_path, "oot_loan_model.csv"))
        self.__product = pd.read_csv(os.path.join(file_input_path, "oot_product_model.csv"))
        self.__model = joblib.load(model_input_path)

        self.__loan_feature = None
        self.__loan_label = None
        self.__loan_proba = None
        self.__loan_score = None

    def predict_loan(self):
        feature_selected = ["trans_num_max_by_date", "succ_trans_num_max_by_shop", "taobaoing_date", "open_last_days",
                            "tb0020003", "tb0020010", "ep0010004", "mp0010009", "ep0030009", "mp0010022",
                            "cnt_call_std", "tb0020004", "ep0060003", "ep0060004"]
        label_selected = ["is_fraud"]

        self.__loan_feature = self.__loan[feature_selected]
        self.__loan_feature = self.__loan_feature.fillna(-999)
        self.__loan_label = self.__loan[label_selected]

        print(roc_auc_score(self.__loan_label, self.__model.predict_proba(self.__loan_feature)[:, 1]))
        print(recall_score(self.__loan_label, self.__model.predict(self.__loan_feature)))

        self.__loan_proba = pd.Series(self.__model.predict_proba(self.__loan_feature)[:, 1])
        self.__loan_score = pd.Series(self.__loan_proba).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x)))

        pd.concat([self.__loan_score.to_frame("score"), self.__loan_label], axis=1).to_csv("C:\\Users\\Dell\\Desktop\\loan_score.csv", index=False)

    def predict_product(self):
        pass


if __name__ == "__main__":
    rfb = RandomForestBaselineOot(file_input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data", model_input_path="C:\\Users\\Dell\\Desktop\\week\FC\\anti_fraud\\model\\rf.z")
    rfb.predict_loan()

