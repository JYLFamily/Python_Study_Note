import os
import numpy as np
import pandas as pd

def fread():
    path = ""
    taobao_01 = pd.read_csv(os.path.join(path, ""), \
                            sep="\t", header=None)
    creditcard_01 = pd.read_csv(os.path.join(path, ""), \
                                sep="\t", header=None)
    zhengxin_01 = pd.read_csv(os.path.join(path, ""), \
                              sep="\t", header=None)
    yunyingshang_01 = pd.read_csv(os.path.join(path, ""), \
                                  sep="\t", header=None)

    return taobao_01, creditcard_01, zhengxin_01, yunyingshang_01


def rename_features(list_features):
    list_names = []
    list_features_return = []

    for name, feature in zip(list_names, list_features):
        feature.columns = ["0"] + [name + "_" + str(i) for i in feature.columns if i > 0]
        list_features_return.append(feature)

    return list_features_return


def merge_features(list_features):
    first_feature = list_features[0]
    second_feature = list_features[1]
    temp = first_feature.merge(second_feature, left_on=["0"], right_on=["0"], how="inner")

    for item_feature in list_features[2:]:
        temp = temp.merge(item_feature, left_on=["0"], right_on=["0"], how="inner")

    temp = temp.rename(columns={"0":"apply_id_no"})

    return temp

def select_features(raw_data):
    raw_data = raw_data.loc[:, []]

    return raw_data

def fwrite(feature):
    feature.to_csv("C:\\Users\\Dell\\Desktop\\all_mini_05", sep="\t", header=False, index=False)


if __name__ == "__main__":
    taobao_05, creditcard_05, zhengxin_05, yunyingshang_05 = fread()
    taobao_05, creditcard_05, zhengxin_05, yunyingshang_05 = \
        rename_features([taobao_05, creditcard_05, zhengxin_05, yunyingshang_05])

    all_05 = merge_features([taobao_05, creditcard_05, zhengxin_05, yunyingshang_05])
    all_05 = select_features(all_05)
    fwrite(all_05)
