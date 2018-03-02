# coding:utf-8

from Project.LostRepair.StackingV3.FeatureEngineering.FeatureBinningWoe import *
from Project.LostRepair.StackingV3.FeatureEngineering.FeatureBinningUtil import *
from sklearn.preprocessing import LabelEncoder


import numpy as np


def fb_categorical_linear_model(train_categorical, test_categorical, train_label):
    col_header = train_categorical.columns

    for col in col_header:
        # 5 个水平以上
        if len(np.unique(train_categorical[col])) > 5:
            train_categorical[col], test_categorical[col] = bad_rate_transform(train_categorical, test_categorical, col, train_label)
            train_flag, test_flag = fb_numeric_linear_model(train_categorical[col], test_categorical[col], train_label, col)
            if not train_flag:
                train_categorical.drop(labels=[col], axis=1, inplace=True)
                test_categorical.drop(labels=[col], axis=1, inplace=True)
            else:
                train_categorical[col] = train_flag
                test_categorical[col] = test_flag
        # 4、5 个水平
        elif len(np.unique(train_categorical[col])) > 3:
            train_categorical[col], test_categorical[col] = bad_rate_transform(train_categorical, test_categorical, col, train_label)
            train_flag, test_flag = fb_numeric_linear_model(train_categorical[col], test_categorical[col], train_label, col)
            if not train_flag:
                train_categorical.drop(labels=[col], axis=1, inplace=True)
                test_categorical.drop(labels=[col], axis=1, inplace=True)
            else:
                train_categorical[col] = train_flag
                test_categorical[col] = test_flag
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
        max_bin = 5 if len(np.unique(train_numeric)) > 5 else 3
        level_to_bin = chi_merge_bin(train_numeric, col, train_label, max_bin)
        train_numeric_new = train_numeric.map(lambda x: value_to_bin(x, level_to_bin))

        if max_bin == 5:
            # 是否单调 , 不单调 max_bin-1 重新进行卡方分箱
            if not bad_rate_monotone(train_numeric, col, train_label):
                for max_bin in range(4, 1, -1):
                    level_to_bin = chi_merge_bin(train_numeric, col, train_label, max_bin)
                    train_numeric[col] = train_numeric[col].map(lambda x: value_to_bin(x, level_to_bin))
                    if bad_rate_monotone(train_numeric, col, train_label):
                        break
            # 是否存在一箱样本占比超过 90% , 如果超过删除这个特征
            if maximum_bin_pcnt(train_numeric, col, train_label) > 0.9:
                return False, False
            # 是否存在一箱样本没有 bad , 如果存在与 bad rate 次低的箱 merge , merge 后需要再次检查是否有一箱样本占比超过 90%
            if minimum_bad_bin_pcnt(train_numeric, col, train_label) == 0.0:
                train_numeric[col] = train_numeric[col].map(merge_bad_bin(train_numeric, col, train_label))
                if maximum_bin_pcnt(train_numeric, col, train_label) > 0.9:
                    return False, False
        else:
            # 是否存在一箱样本占比超过 90% , 如果超过删除这个特征
            if maximum_bin_pcnt(train_numeric, col, train_label) > 0.9:
                return False, False
            # 是否存在一箱样本没有 bad , 如果存在与 bad rate 次低的箱 merge , merge 后需要再次检查是否有一箱样本占比超过 90%
            if minimum_bad_bin_pcnt(train_numeric, col, train_label) == 0.0:
                train_numeric[col] = train_numeric[col].map(merge_bad_bin(train_numeric, col, train_label))
                if maximum_bin_pcnt(train_numeric, col, train_label) > 0.9:
                    return False, False

        test_numeric_new = test_numeric.map(lambda x: value_to_bin(x, level_to_bin))

        return train_numeric_new, test_numeric_new
    else:
        col_header = train_numeric.columns

        for col in col_header:
            level_to_bin = chi_merge_bin(train_numeric, col, train_label)
            train_numeric[col] = train_numeric[col].map(lambda x: value_to_bin(x, level_to_bin))
            # 是否单调 , 不单调 max_bin-1 重新进行卡方分箱
            if not bad_rate_monotone(train_numeric, col, train_label):
                for max_bin in range(4, 1, -1):
                    level_to_bin = chi_merge_bin(train_numeric, col, train_label, max_bin)
                    train_numeric[col] = train_numeric[col].map(lambda x: value_to_bin(x, level_to_bin))
                    if bad_rate_monotone(train_numeric, col, train_label):
                        break
            # # 是否存在一箱样本占比超过 90% , 如果超过删除这个特征
            # if maximum_bin_pcnt(train_numeric, col, train_label) > 0.9:
            #     train_numeric.drop(labels=[col], axis=1, inplace=True)
            #     test_numeric.drop(labels=[col], axis=1, inplace=True)
            #     continue
            # # 是否存在一箱样本没有 bad , 如果存在与 bad rate 次低的箱 merge , merge 后需要再次检查是否有一箱样本占比超过 90%
            # if minimum_bad_bin_pcnt(train_numeric, col, train_label) == 0.0:
            #     train_numeric[col] = train_numeric[col].map(merge_bad_bin(train_numeric, col, train_label))
            #     if maximum_bin_pcnt(train_numeric, col, train_label) > 0.9:
            #         train_numeric.drop(labels=[col], axis=1, inplace=True)
            #         test_numeric.drop(labels=[col], axis=1, inplace=True)
            #         continue
            # woe 转换
            test_numeric[col] = test_numeric[col].map(lambda x: value_to_bin(x, level_to_bin))
            train_numeric[col], test_numeric[col] = woe_transform(train_numeric, test_numeric, col, train_label)

        return train_numeric, test_numeric


def fb_numeric_tree_model():
    pass


if __name__ == "__main__":
    train_feature = pd.read_csv("C:\\Users\\Dell\\Desktop\\new_train.csv", encoding="gbk")
    train_label = pd.read_csv("C:\\Users\\Dell\\Desktop\\new_train.csv", usecols=[4], encoding="gbk")
    test_feature = pd.read_csv("C:\\Users\\Dell\Desktop\\test.csv", encoding="gbk")
    test_label = pd.read_csv("C:\\Users\\Dell\Desktop\\test.csv", encoding="gbk")

    train_belong_province = train.loc[:, "belong_province"].astype(str)
    test_belong_province = test.loc[:, "belong_province"].astype(str)
    le = LabelEncoder().fit(train_belong_province)
    train_belong_province = le.transform(train_belong_province.to_series())

    test_belong_province = le.transform(test_belong_province.to_series())
    fb_categorical_linear_model(train_belong_province.to_frame(), )

    train = (train[["open_last_days", "integral", "account_balance", "cnt_called",
                    "cnt_calling", "communication_duration_called",
                    "communication_duration_calling", "natural_contact"]])
    test = (test[["user_label", "open_last_days", "integral", "account_balance", "cnt_called",
                  "cnt_calling", "communication_duration_called",
                  "communication_duration_calling", "natural_contact"]])

    train_new, test_new = fb_numeric_linear_model(train, test, train_label=train_label)
    pd.concat([train_label, train_new], axis=1).to_csv("C:\\Users\\Dell\\Desktop\\train_chiq.csv", index=False)
    test_new.to_csv("C:\\Users\\Dell\\Desktop\\test_chiq.csv", index=False)