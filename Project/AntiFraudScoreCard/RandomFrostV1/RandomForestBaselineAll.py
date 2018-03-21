# coding:utf-8

import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib


class RandomForestBaselineAll(object):

    def __init__(self, file_input_path, model_input_path):
        self.__file_input_path = file_input_path
        self.__model_input_path = model_input_path
        self.__all = pd.read_csv(os.path.join(file_input_path, "fc_qianzhan_product_anti_fraud_summary_product.csv"))
        self.__model = joblib.load(model_input_path)

        self.__predict_label = None
        self.__predict_proba = None
        self.__predict_score = None

    def predict_all(self):
        feature_selected = ["trans_num_max_by_date", "succ_trans_num_max_by_shop", "taobaoing_date", "open_last_days",
                            "tb0020003", "tb0020010", "ep0010004", "mp0010009", "ep0030009", "mp0010022",
                            "cnt_call_std", "tb0020004", "ep0060003", "ep0060004"]

        self.__all = self.__all[feature_selected]
        self.__all = self.__all.fillna(-999)
        self.__predict_label = pd.Series(self.__model.predict(self.__all.values)).to_frame("label")
        self.__predict_proba = pd.Series(self.__model.predict_proba(self.__all.values)[:, 1])
        self.__predict_score = self.__predict_proba.apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x))).to_frame("score")

        pd.concat([self.__predict_score, self.__predict_label], axis=1).to_csv(os.path.join(self.__file_input_path, "anti_fraud_score_label.csv"), index=False)


if __name__ == "__main__":
    rfb = RandomForestBaselineAll(file_input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data", model_input_path="C:\\Users\\Dell\\Desktop\\week\FC\\anti_fraud\\model\\rf.pkl")
    rfb.predict_all()