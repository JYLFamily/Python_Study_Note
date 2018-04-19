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

        self.__dtc = None
        self.__dot = None
        self.__graph = None

    def merge_df(self):
        self.__score_label_df = self.__score_df.merge(self.__label_df, left_on=["id_no"], right_on=["id_no"], how="inner")
        self.__score_label_df = self.__score_label_df.loc[(self.__score_label_df["user_label"].notnull()), :]
        self.__score_label_df = self.__score_label_df.loc[(self.__score_label_df["user_label"] != 2), :]
        self.__score_label_df["user_label"] = self.__score_label_df["user_label"].apply(lambda x: 0 if x == 1 else 1)

    def train_model(self):
        self.__dtc = DecisionTreeClassifier(max_depth=3)
        self.__dtc.fit(self.__score_label_df[["anti_fraud_score", "combined_scores_02"]], self.__score_label_df["user_label"])

    def stat_model(self):
        pass

    def viz_model(self):
        self.__dot = export_graphviz(
            self.__dtc,
            out_file=None,
            feature_names=["anti_fraud_score", "combined_scores_02"],
            class_names=["bad", "good"],
            filled=True,     # 二分类问题, 该叶子节点 majority class
            node_ids=True,   # 树模型节点生成顺序
            proportion=True, # 落在该叶子节点的样本比例
            rounded=True     # 绘图相关
        )
        self.__graph = pydotplus.graph_from_dot_data(self.__dot)
        self.__graph.write_pdf("C:\\Users\\Dell\\Desktop\\graph.pdf")


if __name__ == "__main__":
    score_df_path = "C:\\Users\\Dell\\Desktop\\week\\FC\\Threshold\\2018_04_08\\data\\ma_score_new.csv"
    label_df_path = "C:\\Users\\Dell\\Desktop\\week\\FC\\score_card\\yunyingshang\\before\\data\\fc_qianzhan_product_sample_user_feature_mobile.csv"
    score_df_use_cols = ["id_no", "grade_of_limit", "mobile_score_1", "taobao_score", "mobile_score_2", "anti_fraud_score", "combined_scores_01", "combined_scores_02"]
    label_df_use_cols = ["id_no", "user_label"]

    ft = FirstTry(score_df_path=score_df_path, label_df_path=label_df_path, score_df_use_cols=score_df_use_cols, label_df_use_cols=label_df_use_cols)
    ft.merge_df()
    ft.train_model()
    ft.stat_model()
    ft.viz_model()