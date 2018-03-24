# coding:utf-8

import os
import pandas as pd
from collections import Counter


class SetUserLevel(object):
    def __init__(self, *, input_path):
        self.__input_path = input_path
        self.__train_feature = pd.read_csv(os.path.join(input_path, "train_feature.csv"), encoding="gbk")
        self.__train_label = []
        self.__validation_feature = pd.read_csv(os.path.join(input_path, "validation_feature.csv"), encoding="gbk")
        self.__validation_label = []

    def set_label(self):

        for i in range(self.__train_feature.shape[0]):
            if self.__train_feature.iloc[i]["status"] == "COMMON_REPAY" and self.__train_feature.iloc[i]["real_amount"] <= 3000:
                self.__train_label.append(0)
            elif self.__train_feature.iloc[i]["status"] == "COMMON_REPAY" and self.__train_feature.iloc[i]["real_amount"] <= 5000:
                self.__train_label.append(1)
            elif self.__train_feature.iloc[i]["status"] == "COMMON_REPAY" and self.__train_feature.iloc[i]["real_amount"] > 5000:
                self.__train_label.append(2)
            elif self.__train_feature.iloc[i]["status"] == "OVERDUE_REPAY" and self.__train_feature.iloc[i]["real_amount"] <= 3000:
                self.__train_label.append(3)
            elif self.__train_feature.iloc[i]["status"] == "OVERDUE_REPAY" and self.__train_feature.iloc[i]["real_amount"] <= 5000:
                self.__train_label.append(4)
            elif self.__train_feature.iloc[i]["status"] == "OVERDUE_REPAY" and self.__train_feature.iloc[i]["real_amount"] > 5000:
                self.__train_label.append(5)
            elif self.__train_feature.iloc[i]["status"] == "OVERDUE" and self.__train_feature.iloc[i]["real_amount"] <= 3000:
                self.__train_label.append(6)
            elif self.__train_feature.iloc[i]["status"] == "OVERDUE" and self.__train_feature.iloc[i]["real_amount"] <= 5000:
                self.__train_label.append(7)
            elif self.__train_feature.iloc[i]["status"] == "OVERDUE" and self.__train_feature.iloc[i]["real_amount"] > 5000:
                self.__train_label.append(8)

        for i in range(self.__validation_feature.shape[0]):
            if self.__validation_feature.iloc[i]["status"] == "COMMON_REPAY" and self.__validation_feature.iloc[i]["real_amount"] <= 3000:
                self.__validation_label.append(0)
            elif self.__validation_feature.iloc[i]["status"] == "COMMON_REPAY" and self.__validation_feature.iloc[i]["real_amount"] <= 5000:
                self.__validation_label.append(1)
            elif self.__validation_feature.iloc[i]["status"] == "COMMON_REPAY" and self.__validation_feature.iloc[i]["real_amount"] > 5000:
                self.__validation_label.append(2)
            elif self.__validation_feature.iloc[i]["status"] == "OVERDUE_REPAY" and self.__validation_feature.iloc[i]["real_amount"] <= 3000:
                self.__validation_label.append(3)
            elif self.__validation_feature.iloc[i]["status"] == "OVERDUE_REPAY" and self.__validation_feature.iloc[i]["real_amount"] <= 5000:
                self.__validation_label.append(4)
            elif self.__validation_feature.iloc[i]["status"] == "OVERDUE_REPAY" and self.__validation_feature.iloc[i]["real_amount"] > 5000:
                self.__validation_label.append(5)
            elif self.__validation_feature.iloc[i]["status"] == "OVERDUE" and self.__validation_feature.iloc[i]["real_amount"] <= 3000:
                self.__validation_label.append(6)
            elif self.__validation_feature.iloc[i]["status"] == "OVERDUE" and self.__validation_feature.iloc[i]["real_amount"] <= 5000:
                self.__validation_label.append(7)
            elif self.__validation_feature.iloc[i]["status"] == "OVERDUE" and self.__validation_feature.iloc[i]["real_amount"] > 5000:
                self.__validation_label.append(8)

    def output_label(self):
        # print(Counter(self.__train_label))
        pd.Series(self.__train_label).to_frame("label").to_csv(os.path.join(self.__input_path, "train_label.csv"), index=False)
        pd.Series(self.__validation_label).to_frame("label").to_csv(os.path.join(self.__input_path, "validation_label.csv"), index=False)


if __name__ == "__main__":
    sul = SetUserLevel(input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\user_level\\data")
    sul.set_label()
    sul.output_label()