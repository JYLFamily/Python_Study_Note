# coding:utf-8

from Project.LostRepair.StackingV3.FeatureEngineering.FeatureBinningWoe import *
from Project.LostRepair.StackingV3.FeatureEngineering.FeatureBinningUtil import *


import numpy as np


def fb_categorical_linear_model(train_categorical, test_categorical, train_label):
    col_header = train_categorical.columns

    for col in col_header:
        # 5 个水平以上
        if len(np.unique(train_categorical[col])) > 5:
            train_categorical[col], test_categorical[col] = encode_bad_rate(train_categorical, test_categorical, col, train_label)
            train_categorical[col], test_categorical[col] = fb_numeric_linear_model(
                train_categorical[col],
                test_categorical[col],
                train_label,
                col
            )
        # 4、5 个水平
        elif len(np.unique(train_categorical[col])) > 3:
            train_categorical[col], test_categorical[col] = encode_bad_rate(train_categorical, test_categorical, col, train_label)
            train_categorical[col], test_categorical[col] = fb_numeric_linear_model(
                train_categorical[col],
                test_categorical[col],
                train_label,
                col
            )
        # 小于等于 3 个水平
        else:
            if maximum_bin_pcnt(train_categorical, col, train_label) > 0.9:
                train_categorical.drop(labels=[col], axis=1, inplace=True)
                test_categorical.drop(labels=[col], axis=1, inplace=True)
                continue
            if minimum_bad_bin_pcnt(train_categorical, col, train_label) == 0.0:
                train_categorical[col] = train_categorical[col].map(merge_bad_bin(train_categorical, col, train_label))
                test_categorical[col] = test_categorical[col].map(merge_bad_bin(train_categorical, col, train_label))
                if maximum_bin_pcnt(train_categorical, col, train_label) > 0.9:
                    train_categorical.drop(labels=[col], axis=1, inplace=True)
                    test_categorical.drop(labels=[col], axis=1, inplace=True)
                    continue
            train_categorical[col], test_categorical[col] = woe_transform(train_categorical, test_categorical, col, train_label)

    return train_categorical, test_categorical


def fb_categorical_tree_model():
    pass


def fb_numeric_linear_model(train_numeric, test_numeric, train_label, col=None):
    if col is not None:
        max_interval = 5 if len(np.unique(train_numeric)) > 5 else 3
        level_to_bin = chi_merge_bin(train_numeric, col, train_label, max_interval)
        train_numeric_new = train_numeric.map(lambda x: value_to_bin(x, level_to_bin))
        test_numeric_new = test_numeric.map(lambda x: value_to_bin(x, level_to_bin))

        return train_numeric_new, test_numeric_new
    else:
        col_header = train_numeric.columns

        for col in col_header:
            level_to_bin = chi_merge_bin(train_numeric, col, train_label)
            train_numeric[col] = train_numeric[col].map(lambda x: value_to_bin(x, level_to_bin))

            if not bad_rate_monotone(train_numeric, col, train_label):
                for max_interval in range(4, 1, -1):
                    level_to_bin = chi_merge_bin(train_numeric, col, train_label, max_interval)
                    train_numeric[col] = train_numeric[col].map(lambda x: value_to_bin(x, level_to_bin))
                    if bad_rate_monotone(train_numeric, col, train_label):
                        break

            if maximum_bin_pcnt(train_numeric, col, train_label) > 0.9:
                train_numeric.drop(labels=[col], axis=1, inplace=True)
                test_numeric.drop(labels=[col], axis=1, inplace=True)
                continue

            train_numeric[col], test_numeric[col] = woe_transform(train_numeric, test_numeric, col, train_label)

        return train_numeric, test_numeric


def fb_numeric_tree_model():
    pass


if __name__ == "__main__":
    train = pd.read_csv(
        "C:\\Users\\Dell\\Desktop\\new_train.csv",
        encoding="gbk"
    )
    train_label = pd.read_csv(
        "C:\\Users\\Dell\\Desktop\\new_train.csv",
        usecols=[4],
        encoding="gbk"
    )
    test = pd.read_csv(
        "C:\\Users\\Dell\Desktop\\test.csv",
        encoding="gbk"
    )

    train = train[["open_last_days", "integral", "account_balance", "cnt_called", "cnt_calling", "communication_duration_called", "communication_duration_calling", "natural_contact"]]
    test = test[["user_label", "open_last_days", "integral", "account_balance", "cnt_called", "cnt_calling", "communication_duration_called", "communication_duration_calling", "natural_contact"]]
    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)
    fb_numeric_linear_model(train, test, train_label=train_label)