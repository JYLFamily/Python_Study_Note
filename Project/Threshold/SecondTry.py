# coding:utf-8

import pydotplus
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


class FirstTry(object):
    def __init__(self, *, score_df_path, label_df_path, score_df_use_cols, label_df_use_cols):
        self.__score_df = pd.read_csv(score_df_path, usecols=score_df_use_cols)
        self.__label_df = pd.read_csv(label_df_path, usecols=label_df_use_cols)
        self.__score_label_df = None

    def merge_df(self):
        self.__score_label_df = self.__score_df.merge(self.__label_df, left_on=["id_no"], right_on=["id_no"], how="inner")
        self.__score_label_df = self.__score_label_df.loc[(self.__score_label_df["user_label"].notnull()), :]
        self.__score_label_df = self.__score_label_df.loc[(self.__score_label_df["user_label"] != 2), :]
        self.__score_label_df["user_label"] = self.__score_label_df["user_label"].apply(lambda x: 0 if x == 1 else 1)

    def state_df(self):
        self.__score_label_df["anti_fraud_score"] = pd.qcut(self.__score_label_df["anti_fraud_score"], 10)

        mini_group_01_bin = pd.DataFrame()
        mini_group_01_bad_rate = pd.DataFrame()
        mini_group_01_user_level = pd.DataFrame()

        mini_group_02_bin = pd.DataFrame()
        mini_group_02_bad_rate = pd.DataFrame()
        mini_group_02_user_level = pd.DataFrame()

        for i in list(np.unique(self.__score_label_df["anti_fraud_score"])):
            mini_group = self.__score_label_df.loc[self.__score_label_df["anti_fraud_score"] == i, :]
            mini_group["combined_scores_01"] = pd.qcut(mini_group["combined_scores_01"], 10)
            mini_group["combined_scores_02"] = pd.qcut(mini_group["combined_scores_02"], 10)
            print(mini_group.groupby(["combined_scores_01", "grade_of_limit"])["grade_of_limit"].count())
            mini_group_01 = pd.concat([
                mini_group.groupby(["combined_scores_01"])["user_label"].count().to_frame("count"),
                mini_group.groupby(["combined_scores_01"])["user_label"].sum().to_frame("sum"),
                mini_group.groupby(["combined_scores_01"])["grade_of_limit"].mean().to_frame("mean")
            ], axis=1)
            mini_group_01["bad rate"] = mini_group_01["sum"] / mini_group_01["count"]
            mini_group_01_bin = pd.concat([mini_group_01_bin, pd.Series(mini_group_01["bad rate"].index).to_frame("bin").reset_index(drop=True)], axis=1)
            mini_group_01_bad_rate = pd.concat([mini_group_01_bad_rate, mini_group_01["bad rate"].to_frame().reset_index(drop=True)], axis=1)
            mini_group_01_user_level = pd.concat([mini_group_01_user_level, mini_group_01["mean"].to_frame().reset_index(drop=True)], axis=1)

            mini_group_02 = pd.concat([
                mini_group.groupby(["combined_scores_02"])["user_label"].count().to_frame("count"),
                mini_group.groupby(["combined_scores_02"])["user_label"].sum().to_frame("sum"),
                mini_group.groupby(["combined_scores_02"])["grade_of_limit"].mean().to_frame("mean")
            ], axis=1)
            mini_group_02["bad rate"] = mini_group_02["sum"] / mini_group_02["count"]
            mini_group_02_bin = pd.concat([mini_group_02_bin, pd.Series(mini_group_02["bad rate"].index).to_frame("bin").reset_index(drop=True)], axis=1)
            mini_group_02_bad_rate = pd.concat([mini_group_02_bad_rate, mini_group_02["bad rate"].to_frame().reset_index(drop=True)], axis=1)
            mini_group_02_user_level = pd.concat([mini_group_02_user_level, mini_group_02["mean"].to_frame().reset_index(drop=True)], axis=1)

        mini_group_01_bin = mini_group_01_bin.sort_index(ascending=False)
        mini_group_01_bad_rate = mini_group_01_bad_rate.sort_index(ascending=False)
        mini_group_01_user_level = mini_group_01_user_level.sort_index(ascending=False)

        mini_group_02_bin = mini_group_02_bin.sort_index(ascending=False)
        mini_group_02_bad_rate = mini_group_02_bad_rate.sort_index(ascending=False)
        mini_group_02_user_level = mini_group_02_user_level.sort_index(ascending=False)


if __name__ == "__main__":
    score_df_path = "C:\\Users\\Dell\\Desktop\\week\\FC\\Threshold\\2018_04_08\\data\\ma_score_new.csv"
    label_df_path = "C:\\Users\\Dell\\Desktop\\week\\FC\\score_card\\yunyingshang\\before\\data\\fc_qianzhan_product_sample_user_feature_mobile.csv"
    score_df_use_cols = ["id_no", "grade_of_limit", "mobile_score_1", "taobao_score", "mobile_score_2", "anti_fraud_score", "combined_scores_01", "combined_scores_02"]
    label_df_use_cols = ["id_no", "user_label"]

    ft = FirstTry(score_df_path=score_df_path, label_df_path=label_df_path, score_df_use_cols=score_df_use_cols, label_df_use_cols=label_df_use_cols)
    ft.merge_df()
    ft.state_df()