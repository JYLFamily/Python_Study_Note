# coding:utf-8

import numpy as np
import pandas as pd


def woe_transform(train_categorical, test_categorical, col, train_label):
    train = pd.DataFrame({col: train_categorical[col], "label": train_label.squeeze()})
    level_to_woe = {}

    total = train.groupby([col])["label"].count()
    total = pd.DataFrame({"total": total})
    bad = train.groupby([col])["label"].sum()
    bad = pd.DataFrame({"bad": bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how="left")
    regroup.reset_index(drop=False, inplace=True)
    regroup["bad_pcnt"] = regroup["bad"] / np.sum(regroup["bad"])
    regroup["good"] = regroup["total"] - regroup["bad"]
    regroup["good_pcnt"] = regroup["good"] / np.sum(regroup["good"])
    regroup["woe"] = np.log(regroup["good_pcnt"] / regroup["bad_pcnt"])

    for i, j in zip(regroup[col], regroup["woe"]):
        level_to_woe[i] = j
    print(level_to_woe)
    return train_categorical[col].map(level_to_woe), test_categorical[col].map(level_to_woe)