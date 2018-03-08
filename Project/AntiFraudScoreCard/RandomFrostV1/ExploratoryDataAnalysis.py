# coding:utf-8

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ExploratoryDataAnalysis(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path
        self.__train_feature = None
        self.__train_label = None
        self.__train = None
        self.__train_mini = None
        self.__categorical_header = []
        self.__numeric_header = []

    def read(self):
        self.__train_feature = pd.read_csv(os.path.join(self.__input_path, "train_feature.csv"), encoding="gbk")
        self.__train_label = pd.read_csv(os.path.join(self.__input_path, "train_label.csv"))

    def set_categorical_and_numeric_header(self):
        column_header = self.__train_feature

        for col in column_header:
            if len(np.unique(self.__train_feature[col])) > 20:
                self.__numeric_header.append(col)
            else:
                self.__categorical_header.append(col)

    def exploratory_numeric_feature(self):
        # for numeric_col in self.__numeric_header:
        #     self.__train = pd.concat([self.__train_feature[numeric_col].to_frame(), self.__train_label], axis=1)
        #     self.__train["if_fraud"] = self.__train["if_fraud"].astype(str)
        #     self.__train[[numeric_col, "if_fraud"]].boxplot(by="if_fraud")
        #     plt.show()

        self.__train = pd.concat([self.__train_feature["cb0180003"].to_frame(), self.__train_label], axis=1)
        self.__train["if_fraud"] = self.__train["if_fraud"].astype(str)

    def exploratory_categorical_feature(self):
        pass


if __name__ == "__main__":
    eda = ExploratoryDataAnalysis(input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data")
    eda.read()
    # eda.set_categorical_and_numeric_header()
    eda.exploratory_numeric_feature()
