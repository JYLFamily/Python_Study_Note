# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FeatureEngineering(object):

    def __init__(self, *, train, train_label, test, test_label):
        self.__train = train
        self.__train_label = train_label
        self.__test = test
        self.__test_label = test_label

        self.__columns = []
        self.__numerical_cloumns = []
        self.__categorical_columns = []

        self.__train_numerical = None
        self.__train_categorical = None
        self.__test_numerical = None
        self.__test_categorical = None

        self.__train_tree = None
        self.__test_tree = None
        self.__train_linear = None
        self.__test_linear = None

    def __set_numerical_and_categorical_cloumns(self):
        self.__columns = self.__train.columns
        for col in self.__columns:
            if len(np.unique(self.__train[col])) > 15:
                self.__numerical_cloumns.append(col)
            else:
                self.__categorical_columns.append(col)

        self.__train_numerical = self.__train[self.__numerical_cloumns]
        self.__train_categorical = self.__train[self.__categorical_columns]
        self.__test_numerical = self.__test[self.__numerical_cloumns]
        self.__test_categorical = self.__test[self.__categorical_columns]

    def tree_model_feature_engineering(self):
        self.__set_numerical_and_categorical_cloumns()

        encoder = LabelEncoder().fit(self.__train_categorical)
        self.__train_categorical = (encoder.transform(self.__train_categorical)
                                    .reshape((-1, self.__train_categorical.shape[1])))
        self.__test_categorical = (encoder.transform(self.__test_categorical)
                                   .reshape((-1, self.__test_categorical.shape[1])))

        self.__train_tree = np.hstack((self.__train_categorical, self.__train_numerical))
        self.__test_tree = np.hstack((self.__test_categorical, self.__test_numerical))

        return self.__train_tree, self.__test_tree

    def linear_model_feature_engineering(self):
        self.__set_numerical_and_categorical_cloumns()

        encoder = OneHotEncoder(sparse=False).fit(self.__train_categorical)
        self.__train_categorical = encoder.transform(self.__train_categorical)
        self.__test_categorical = encoder.transform(self.__test_categorical)

        scaler = StandardScaler().fit(self.__train_numerical)
        self.__train_numerical = scaler.transform(self.__train_numerical)
        self.__test_numerical = scaler.transform(self.__test_numerical)

        self.__train_linear = np.hstack((self.__train_categorical, self.__train_numerical))
        self.__test_linear = np.hstack((self.__test_categorical, self.__test_numerical))

        return self.__train_linear, self.__test_linear


if __name__ == "__main__":
    X = pd.read_csv("C:\\Users\\Dell\\Desktop\\model2.csv", usecols=list(range(1, 4)))
    y = pd.read_csv("C:\\Users\\Dell\\Desktop\\model2.csv", usecols=[0])
    train, test, train_label, test_label = (
        train_test_split(X, y, test_size=0.2, random_state=9))

    fe = FeatureEngineering(train=train, train_label=train_label, test=test, test_label=test_label)
    fe.linear_model_feature_engineering()
    fe.tree_model_feature_engineering()