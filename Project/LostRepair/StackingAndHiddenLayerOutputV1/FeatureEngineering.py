# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split


class FeatureEngineering(object):
    @staticmethod
    def __set_numerical_and_categorical_cloumns(train, test):
        numerical_cloumns = []
        categorical_columns = []

        for col in list(range(train.shape[1])):
            if len(np.unique(train[:, col])) > 15:
                numerical_cloumns.append(col)
            else:
                categorical_columns.append(col)

        train_numerical = train[:, numerical_cloumns]
        train_categorical = train[:, categorical_columns]
        test_numerical = test[:, numerical_cloumns]
        test_categorical = test[:, categorical_columns]

        return train_numerical, train_categorical, test_numerical, test_categorical

    @staticmethod
    def tree_model_feature_engineering(*, train, test):
        train_numerical, train_categorical, test_numerical, test_categorical = (
            FeatureEngineering.__set_numerical_and_categorical_cloumns(train, test))

        # 缺失值填补
        imputer = Imputer(strategy="most_frequent").fit(train_categorical)
        train_categorical = imputer.transform(train_categorical)
        test_categorical = imputer.transform(test_categorical)

        imputer = Imputer(strategy="median").fit(train_numerical)
        train_numerical = imputer.transform(train_numerical)
        test_numerical = imputer.transform(test_numerical)

        # 类别变量一般是字符格式, 需要先使用 LabelEncoder
        for i in range(train_categorical.shape[1]):
            encoder_le = LabelEncoder()
            encoder_le.fit(train_categorical[:, i])
            train_categorical[:, i] = encoder_le.transform(train_categorical[:, i])
            test_categorical[:, i] = encoder_le.transform(test_categorical[:, i])

        train_tree = np.hstack((train_categorical, train_numerical))
        test_tree = np.hstack((test_categorical, test_numerical))

        return train_tree, test_tree

    @staticmethod
    def linear_model_feature_engineering(*, train, test):
        train_numerical, train_categorical, test_numerical, test_categorical = (
            FeatureEngineering.__set_numerical_and_categorical_cloumns(train, test))

        # 缺失值填补
        imputer = Imputer(strategy="most_frequent").fit(train_categorical)
        train_categorical = imputer.transform(train_categorical)
        test_categorical = imputer.transform(test_categorical)

        imputer = Imputer(strategy="median").fit(train_numerical)
        train_numerical = imputer.transform(train_numerical)
        test_numerical = imputer.transform(test_numerical)

        # 类别变量一般是字符格式, 需要先使用 LabelEncoder
        for i in range(train_categorical.shape[1]):
            encoder_le = LabelEncoder()
            encoder_le.fit(train_categorical[:, i])
            train_categorical[:, i] = encoder_le.transform(train_categorical[:, i])
            test_categorical[:, i] = encoder_le.transform(test_categorical[:, i])

        # sparse=False 否则是稀疏格式
        encoder_oht = OneHotEncoder(sparse=False).fit(train_categorical)
        train_categorical = encoder_oht.transform(train_categorical)
        test_categorical = encoder_oht.transform(test_categorical)

        # 数值型特征进行标准化
        scaler = StandardScaler().fit(train_numerical)
        train_numerical = scaler.transform(train_numerical)
        test_numerical = scaler.transform(test_numerical)

        train_linear = np.hstack((train_categorical, train_numerical))
        test_linear = np.hstack((test_categorical, test_numerical))

        return train_linear, test_linear

    @staticmethod
    def net_model_feature_engineering(*, train, test):
        train_net, test_net = (
            FeatureEngineering.linear_model_feature_engineering(train=train,
                                                                test=test)
        )

        return train_net, test_net


if __name__ == "__main__":
    X = pd.read_csv("C:\\Users\\Dell\\Desktop\\model.txt", sep="\t", usecols=list(range(1, 4)))
    y = pd.read_csv("C:\\Users\\Dell\\Desktop\\model.txt", sep="\t", usecols=[0])

    train_X, test_X, train_y, test_y = (
        train_test_split(X, y, test_size=0.2, random_state=9))

    train_X, test_X = (
         FeatureEngineering.net_model_feature_engineering(train=train_X,
                                                          test=test_X)
    )