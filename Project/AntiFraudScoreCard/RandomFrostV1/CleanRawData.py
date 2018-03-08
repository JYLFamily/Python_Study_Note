# coding:utf-8

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CleanRawData(object):

    def __init__(self, *, input_path, output_path):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__raw_df = None
        self.__raw_feature = None
        self.__raw_label = None
        self.__train_feature = None
        self.__train_label = None
        self.__test_feature = None
        self.__test_label = None

    def read(self):
        self.__raw_df = pd.read_csv(self.__input_path, encoding="gbk")

    def clean(self):
        self.__raw_df = self.__raw_df.loc[(np.logical_not(np.isnan(self.__raw_df["if_fraud"]))), :]
        self.__raw_df = self.__raw_df.sort_values(by="create_time")
        self.__raw_label = self.__raw_df["if_fraud"].to_frame("if_fraud")
        self.__raw_feature = self.__raw_df.drop(["id_no", "apply_no", "user_label", "loan_status", "if_fraud"], axis=1)
        self.__train_feature, self.__test_feature, self.__train_label, self.__test_label = train_test_split(
            self.__raw_feature,
            self.__raw_label,
            test_size=0.3,
            shuffle=False
        )

    def write(self):
        self.__train_feature.to_csv(os.path.join(self.__output_path, "train_feature.csv"), index=False)
        self.__test_feature.to_csv(os.path.join(self.__output_path, "test_feature.csv"), index=False)
        self.__train_label.to_csv(os.path.join(self.__output_path, "train_label.csv"), index=False)
        self.__test_label.to_csv(os.path.join(self.__output_path, "test_label.csv"), index=False)


if __name__ == "__main__":
    crd = CleanRawData(
        input_path="C:\\Users\\Dell\\Desktop\\anti_fraud.csv",
        output_path="C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data"
    )
    crd.read()
    crd.clean()
    crd.write()


