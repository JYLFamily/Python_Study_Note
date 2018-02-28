# coding:utf-8

import numpy as np
import pandas as pd


def encode_bad_rate(train_categorical, test_categorical, col, train_label):
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


def maximum_bin_pcnt(train_categorical, col, train_label):
    # train_label 虽然只有一列但是 type 为 DataFrame
    train = pd.DataFrame({col: train_categorical[col], "label": train_label.squeeze()})

    return np.max(train.groupby([col])[col].count() / train.shape[0]) > 0.9


def minimum_bad_bin_pcnt(train_categorical, col, train_label):
    train = pd.DataFrame({col: train_categorical[col], "label": train_label.squeeze()})

    return np.min(train.groupby([col])["label"].sum() / train.groupby([col])["label"].count())


def merge_bad_bin(train_categorical, col, train_label):
    train = pd.DataFrame({col: train_categorical[col], "label": train_label.squeeze()})

    total = train.groupby([col])["label"].count()
    total = pd.DataFrame({"total": total})
    bad = train.groupby([col])["label"].sum()
    bad = pd.DataFrame({"bad": bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how="left")
    regroup.reset_index(drop=False, inplace=True)

    regroup["bad_rate"] = regroup["bad"] / regroup["total"]
    regroup = regroup.sort_values(by="bad_rate")

    level_regroup = [[i] for i in regroup[col]]
    for i in range(regroup.shape[0]):
        level_regroup[1] = level_regroup[0] + level_regroup[1]
        level_regroup.pop(0)
        if regroup["bad_rate"][i+1] > 0.0:
            break

    level_to_bin = {}
    for i in range(len(level_to_bin)):
        for g in level_to_bin[i]:
            level_to_bin[g] = "Bin " + str(i)

    return level_to_bin


def value_to_bin(value, bin):
    if value <= min(bin):
        return min(bin)
    elif value > max(bin):
        return 10e10
    else:
        for i in range(len(bin) - 1):
            if bin[i] < value <= bin[i+1]:
                return bin[i+1]


def chi_merge_bin(train_numeric, col=None, train_label=None, max_interval=5):
    if train_numeric is pd.Series:
        train = pd.DataFrame({col: train_numeric, "label": train_label.squeeze()})
    else:
        train = pd.DataFrame({col: train_numeric[col], "label": train_label.squeeze()})

    level = sorted(list(set(train[col])))
    level_len = len(level)
    if level_len > 100:
        ind_x = [int(i / 100.0 * level_len) for i in range(1, 100)]
        split_x = [level[i] for i in ind_x]
        train[col] = train[col].map(lambda x: value_to_bin(x, split_x))

    total = train.groupby([col])["label"].count()
    total = pd.DataFrame({"total": total})
    bad = train.groupby([col])["label"].sum()
    bad = pd.DataFrame({"bad": bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(drop=False, inplace=True)

    level = sorted(list(set(regroup[col])))
    level_regroup = [[i] for i in level]
    level_regroup_len = len(level_regroup)
    while len(level_regroup) > max_interval:
        chi_list = []
        for level in level_regroup:
            level_regroup_mini = regroup.loc[regroup[col].isin(level), :]
            combined = zip(
                level_regroup_mini["total"].apply(lambda x: x * np.sum(regroup["bad"]) / np.sum(regroup["total"])),
                level_regroup_mini["bad"]
            )
            chi = np.sum([(i[0] - i[1]) ** 2 / i[0] for i in combined])
            chi_list.append(chi)
        min_position = chi_list.index(min(chi_list))

        if min_position == 0:
            combined_position = 1
        elif min_position == level_regroup_len - 1:
            combined_position = min_position - 1
        else:
            if chi_list[min_position - 1] <= chi_list[min_position + 1]:
                combined_position = min_position - 1
            else:
                combined_position = min_position + 1
        level_regroup[min_position] = level_regroup[min_position] + level_regroup[combined_position]
        level_regroup.remove(level_regroup[combined_position])
        level_regroup_len = len(level_regroup)

    level_regroup = [sorted(i) for i in level_regroup]
    level_to_bin = [max(i) for i in level_regroup[:-1]]

    return level_to_bin


def bad_rate_monotone(train_numeric, col, train_label):
    train = pd.DataFrame({col: train_numeric[col], "label": train_label.squeeze()})

    total = train.groupby([col])["label"].count()
    total = pd.DataFrame({"total": total})
    bad = train.groupby([col])["label"].sum()
    bad = pd.DataFrame({"bad": bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how="left")
    regroup.reset_index(drop=False, inplace=True)

    regroup["bad_rate"] = regroup["bad"] / regroup["total"]
    monotone = len(np.unique([regroup["bad_rate"][i] < regroup["bad_rate"][i+1] for i in range(len(regroup["bad_rate"]) - 1)]))

    if monotone == 1:
        return True
    else:
        return False




