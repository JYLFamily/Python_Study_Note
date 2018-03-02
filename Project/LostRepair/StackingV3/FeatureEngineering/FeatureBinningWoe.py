# coding:utf-8

import numpy as np
import pandas as pd


def bad_rate_transform(train_categorical, test_categorical, col, train_label):
    train = pd.DataFrame({col: train_categorical[col], "label": train_label.squeeze()})
    level_to_bad_rate = {}

    total = train.groupby([col])["label"].count()
    total = pd.DataFrame({"total": total})
    bad = train.groupby([col])["label"].sum()
    bad = pd.DataFrame({"bad": bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how="left")
    regroup.reset_index(drop=False, inplace=True)

    regroup["bad_rate"] = regroup["bad"] / regroup["total"]
    for i, j in zip(regroup[col], regroup["bad_rate"]):
        level_to_bad_rate[i] = j

    return train_categorical[col].map(level_to_bad_rate), test_categorical[col].map(level_to_bad_rate)


def woe_transform(train_categorical, test_categorical, col, train_label):
    level_to_woe = {}
    train = pd.DataFrame({col: train_categorical[col], "label": train_label.squeeze()})
    train.fillna(-1, inplace=True)
    train_categorical[col].fillna(-1, inplace=True)
    test_categorical[col].fillna(-1, inplace=True)

    total = train.groupby([col])["label"].count()
    total = pd.DataFrame({"total": total})
    bad = train.groupby([col])["label"].sum()
    bad = pd.DataFrame({"bad": bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how="left")
    regroup.reset_index(drop=False, inplace=True)
    regroup["bad_pcnt"] = regroup["bad"] / np.sum(regroup["bad"])
    regroup["good"] = regroup["total"] - regroup["bad"]
    regroup["good_pcnt"] = regroup["good"] / np.sum(regroup["good"])
    regroup["woe"] = round(np.log(regroup["bad_pcnt"] / regroup["good_pcnt"]), 4)

    for i, j in zip(regroup[col], regroup["woe"]):
        level_to_woe[i] = j
    print(level_to_woe)
    return train_categorical[col].map(level_to_woe), test_categorical[col].map(level_to_woe)