# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression


class FeatureSelectForLostRepair(object):

    def __init__(self, input_path):
        self.__input_path = input_path
        self.__df = None
        # 存储分类变量 DataFrame
        self.__df_label_continuous_variable = None
        # 存储有序变量 FataFrame
        self.__df_label_categorical_variable = None
        self.__df_columns = None
        self.__continuous_variable = []
        self.__categorical_variable = []

    def load_df(self):
        self.__df = pd.read_csv(self.__input_path)

    def arange_feature_df(self):
        self.__df_columns = self.__df.columns
        self.__df_columns = [col for col in self.__df_columns if col != "label"]
        for col in self.__df_columns:
            if len(np.unique(self.__df[col])) > 10:
                self.__continuous_variable.append(col)
            else:
                self.__categorical_variable.append(col)
        self.__continuous_variable.append("label")
        self.__categorical_variable.append("label")
        self.__df_label_continuous_variable = self.__df.loc[:, self.__continuous_variable]
        self.__df_label_categorical_variable = self.__df.loc[:, self.__categorical_variable]

    def select_categorical_variable(self):
        X = (self.__df_label_categorical_variable.loc[:,
             [col for col in self.__df_label_categorical_variable if col != "label"]])
        y = self.__df_label_categorical_variable["label"]

        print(X.columns[SelectKBest(chi2, k=5).fit(X, y).get_support()])

    def select_continuous_variable(self):
        X = (self.__df_label_continuous_variable.loc[:,
             [col for col in self.__df_label_continuous_variable if col != "label"]])
        y = self.__df_label_continuous_variable["label"]
        model_lr = LogisticRegression(penalty='l1')
        print(model_lr.fit(X, y).coef_)



if __name__ == "__main__":
    fsflr = FeatureSelectForLostRepair("C:\\Users\\Dell\\Desktop\\zytsl_robot.csv")
    fsflr.load_df()
    fsflr.arange_feature_df()
    fsflr.select_categorical_variable()
    fsflr.select_continuous_variable()
