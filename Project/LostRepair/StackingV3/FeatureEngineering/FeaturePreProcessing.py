# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Project.LostRepair.StackingV3.FeatureEngineering.FeatureGeneration import *


def fpp_categorical_missing_value_linear_model(train_categorical, test_categorical):
    columns_header = train_categorical.columns
    missing_value_over_half_train = {}
    missing_value_over_half_test = {}

    for col in columns_header:
        # 如果没有缺失值
        if (np.sum(train_categorical[col].isnull()) + np.sum(test_categorical[col])) == 0:
            continue
        # 如果缺失值 > 0.5 删除原始变量使用指示变量
        if (np.sum(train_categorical[col].isnull()) / train_categorical[col].shape[0]) > 0.5:
            missing_value_over_half_train[col] = train_categorical[col]
            train_categorical.drop(col, axis=1, inplace=True)
            missing_value_over_half_test[col] = test_categorical[col]
            test_categorical.drop(col, axis=1, inplace=True)
        # 如果缺失值 < 0.5 使用训练集众数填充
        else:
            # train_categorical[col].mode() 返回一个 series
            train_categorical[col].fillna(train_categorical[col].mode()[0], inplace=True)
            test_categorical[col].fillna(train_categorical[col].mode()[0], inplace=True)
    # 指示变量
    missing_value_over_half_train_df = pd.DataFrame(missing_value_over_half_train).isnull().astype(int)
    missing_value_over_half_test_df = pd.DataFrame(missing_value_over_half_test).isnull().astype(int)

    train_categorical_new = pd.concat([train_categorical, missing_value_over_half_train_df], axis=1)
    test_categorical_new = pd.concat([test_categorical, missing_value_over_half_test_df], axis=1)

    return train_categorical_new, test_categorical_new


def fpp_categorical_missing_value_tree_model():
    pass


def fpp_categorical_linear_model(train_categorical, test_categorical):
    columns_header_categorical = train_categorical.columns
    train_categorical_distribution = {}
    test_categorical_distribution = {}
    train_categorical_unique_count = []

    # 得到分类变量分布特征
    for col in columns_header_categorical:
        encoding = train_categorical.groupby(col).size() / train_categorical.shape[0]
        train_categorical_distribution[col + "Distribution"] = train_categorical[col].map(encoding)
        test_categorical_distribution[col + "Distribution"] = test_categorical[col].map(encoding)

    train_categorical_distribution = pd.DataFrame(train_categorical_distribution)
    test_categorical_distribution = pd.DataFrame(test_categorical_distribution)

    # 分类变量 one hot encoder
    for col in columns_header_categorical:
        train_categorical_unique_count.append(len(np.unique(train_categorical[col])))
    # one hot encoder 会生成冗余变量 , 删掉
    ohc = OneHotEncoder(sparse=False).fit(train_categorical)
    train_categorical_new = (
        np.delete(ohc.transform(train_categorical), np.array(train_categorical_unique_count).cumsum() - 1, axis=1)
    )
    test_categorical_new = (
        np.delete(ohc.transform(test_categorical), np.array(train_categorical_unique_count).cumsum() - 1, axis=1)
    )

    return train_categorical_new, test_categorical_new, train_categorical_distribution.values, test_categorical_distribution.values


def fpp_categorical_tree_model():
    pass


def fpp_numeric_missing_value_linear_model(train_numeric, test_numeric):
    imputer = Imputer(strategy="median").fit(train_numeric)
    train_numeric_new = imputer.transform(train_numeric)
    test_numeric_new = imputer.transform(test_numeric)

    return train_numeric_new, test_numeric_new


def fpp_numeric_missing_value_tree_model():
    pass


def fpp_numeric_linear_model(train_numeric, test_numeric):
    train_numeric = np.log(1 + train_numeric)
    test_numeric = np.log(1 + test_numeric)

    scaler = MinMaxScaler().fit(train_numeric)
    train_numeric_new = scaler.transform(train_numeric)
    test_numeric_new = scaler.transform(test_numeric)

    return train_numeric_new, test_numeric_new


def fpp_numeric_tree_model():
    pass


if __name__ == "__main__":
    X = pd.read_csv(
        "D:\\Code\\Python\\Python_Study_Note\\Project\\LostRepair\\StackingV3\\mtcars.csv",
        usecols=[i for i in range(11) if i != 8]
    )
    y = pd.read_csv(
        "D:\\Code\\Python\\Python_Study_Note\\Project\\LostRepair\\StackingV3\\mtcars.csv",
        usecols=[8]
    )
    train, test, train_label, test_label = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=9)
    train_categorical, test_categorical = fpp_categorical_missing_value_linear_model(
        train[["cyl", "vs"]],
        test[["cyl", "vs"]]
    )
    train_categorical, test_categorical = fg_categorical_categorical(train_categorical, test_categorical)
    fpp_categorical_linear_model(train_categorical, test_categorical)