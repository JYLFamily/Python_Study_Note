# coding:utf-8

import pydotplus
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score
from sklearn.externals.six import StringIO


class DtTry(object):

    def __init__(self, input_path):
        self.__df = pd.read_csv(input_path)
        self.__df_X = None
        self.__df_y = None
        self.__df_train = None
        self.__df_train_label = None
        self.__df_test = None
        self.__df_test_label = None
        self.__df_columns = None
        self.__model_dt = None
        self.__y_prob = None
        self.__y_pred = None
        self.__dot_data = None

    def label_encoder(self):
        self.__df_columns = self.__df.columns
        for col in self.__df_columns:
            self.__df[col] = LabelEncoder().fit_transform(self.__df[col])

    def set_df_X(self):
        self.__df_X = self.__df.loc[:, [col for col in self.__df_columns if col != "class"]].values

    def set_df_y(self):
        self.__df_y = self.__df.loc[:, "class"].values

    def df_split(self):
        self.__df_train, self.__df_test, self.__df_train_label, self.__df_test_label = (train_test_split(self.__df_X,
            self.__df_y, test_size=0.2, random_state=True))

    def model_train(self):
        self.__model_dt = DecisionTreeClassifier().fit(self.__df_train, self.__df_train_label)

    def model_predict(self):
        self.__y_prob = self.__model_dt.predict(self.__df_test)

    def model_eval(self):
        print(roc_auc_score(self.__df_test_label, self.__y_prob))

    def model_viz(self):
        self.__dot_data = StringIO()
        export_graphviz(self.__model_dt, out_file=self.__dot_data)
        graph = pydotplus.graph_from_dot_data(self.__dot_data.getvalue())
        graph.write_png("D:\\Code\\tree.png")


if __name__ == "__main__":
    dt_try = DtTry(input_path="D:\\Code\\Python\\Python_Study_Note\\ML\\xgboost\\mushrooms.csv")
    dt_try.label_encoder()
    dt_try.set_df_X()
    dt_try.set_df_y()
    dt_try.df_split()
    dt_try.model_train()
    dt_try.model_predict()
    dt_try.model_eval()
    dt_try.model_viz()