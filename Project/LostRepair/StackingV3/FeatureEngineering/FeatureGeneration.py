# coding:utf-8

import itertools
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split
from Project.LostRepair.StackingV3.FeatureEngineering.FeaturePreProcessing import *


def fg_categorical_categorical(train_categorical, test_categorical):
    # 列号到列名的映射
    # index_to_name = dict(zip(
    #     list(range(len(train_categorical.columns))),
    #     list(train_categorical.columns)
    # ))

    # 用常量代替 , 因为后续 train_categorical.columns 是会变化的
    columns_length = len(train_categorical.columns)
    columns_header = list(train_categorical.columns)
    for r in range(2, columns_length+1):
        # itertools.combinations column 的组合
        for combination in itertools.combinations(columns_header, r):
            train_categorical["_".join(combination)] = train_categorical[combination[0]].astype(str)
            test_categorical["_".join(combination)] = test_categorical[combination[0]].astype(str)
            for c in combination[1:]:
                train_categorical["_".join(combination)] = \
                    train_categorical["_".join(combination)] + "_" + train_categorical[c].astype(str)
                test_categorical["_".join(combination)] = \
                    test_categorical["_".join(combination)] + "_" + test_categorical[c].astype(str)

    # 对每一列使用 LabelEncoder 但是有个问题如上这种生成分类变量的方式容易使得测试集中某分类变量的水平测试集中没有出现报错
    mapper = DataFrameMapper([(i, LabelEncoder()) for i in train_categorical.columns])
    mapper.fit(train_categorical)
    train_categorical_new = pd.DataFrame(mapper.transform(train_categorical), columns=train_categorical.columns)
    test_categorical_new = pd.DataFrame(mapper.transform(test_categorical), columns=test_categorical.columns)

    return train_categorical_new, test_categorical_new


def fg_categorical_numeric(train_categorical, train_numeric, test_categorical, test_numeric):
    train = pd.concat([train_categorical, train_numeric], axis=1)
    test = pd.concat([test_categorical, test_numeric], axis=1)
    columns_header_categorical = train_categorical.columns
    columns_header_numeric = train_numeric.columns

    for cate_col in columns_header_categorical:
        for numer_col in columns_header_numeric:
            for operator_name, operator_function in zip(["max", "median", "min"], [np.max, np.median, np.min]):
                # train
                name = cate_col + "G" + numer_col + "U" + operator_name
                train_temp = train.groupby([cate_col])[numer_col].agg([operator_function]).reset_index()
                train_temp.columns = [cate_col, name]
                train_numeric[name] = train.merge(train_temp, left_on=[cate_col], right_on=[cate_col], how="left")[name]
                # test
                test_temp = test.groupby([cate_col])[numer_col].agg([operator_function]).reset_index()
                test_temp.columns = [cate_col, name]
                test_numeric[name] = test.merge(test_temp, left_on=[cate_col], right_on=[cate_col], how="left")[name]

    return train_numeric.values, test_numeric.values


def fg_numeric_numeric():
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
        train[["cyl", "vs", "gear"]],
        test[["cyl", "vs", "gear"]]
    )

    train_categorical, test_categorical = fg_categorical_categorical(train_categorical, test_categorical)
    fg_categorical_numeric(train_categorical, train[["drat", "hp"]], test_categorical, test[["drat", "hp"]])
