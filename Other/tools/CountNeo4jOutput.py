# coding:utf-8

import sys
import json
import pandas as pd


class CountNeo4jOutput(object):

    def __init__(self, sep, header, input_path):
        self.sep = sep
        self.header = header
        self.input_path = input_path
        self.input_df = pd.DataFrame

    def resolve_header(self):
        # header 以一个 JSON 串的形式传入 , 解析 JSON 串得到 header 的 list
        self.header = list(json.loads(self.header).values())

    def load_data(self):
        self.input_df = pd.read_csv(self.input_path, sep=self.sep, names=self.header)

    def add_month(self):
        self.input_df["month"] = self.input_df["apply_id_no"].map(lambda x: x[8])

    def modify_touch(self):
        self.input_df["touch"] = self.input_df["touch"].map(lambda x: 1 if x > 0 else 0)

    def count_output(self):
        # DPC 触碰进件数
        jj_label = self.input_df.loc[:, ["month"]].groupby(["month"])["month"].count().values
        # DPC 触碰且规则触碰进件数
        jj_label_touch = self.input_df.loc[(self.input_df["touch"] > 0), ["month", "touch"]] \
            .groupby(["month", "touch"])["touch"].count().values

        # DPC 触碰放款件数
        fk_label = self.input_df.loc[(self.input_df["due"] != -1), ["month"]].groupby(["month"])["month"].count().values
        # DPC 触碰且规则触碰放款件数
        fk_label_touch = (self.input_df.loc[(self.input_df["touch"] > 0) & (self.input_df["due"] != -1), ["month", "touch"]]
                                       .groupby(["month", "touch"])["touch"].count().values)

        # DPC 触碰 M1+
        fk_label_m1plus = (self.input_df.loc[(self.input_df["due"] != -1) & (self.input_df["due"] == 1), ["month"]]
                                        .groupby(["month"])["month"].count().values)

        # DPC 触碰且规则触碰 M1+
        fk_label_touch_m1plus = (self.input_df.loc[(self.input_df["touch"] > 0) & (self.input_df["due"] != -1) & (self.input_df["due"] == 1), ["month", "touch"]]
                                              .groupby(["month", "touch"])["touch"].count())

        print("DPC 打标进件规则触碰率：" + str(jj_label_touch / jj_label))
        print("DPC 打标放款件规则触碰率：" + str(fk_label_touch / fk_label))
        print("DPC M1+ ：" + str(fk_label_m1plus / fk_label))
        print("------------------------------------")
        print("规则触碰进件量：" + str(jj_label_touch))
        print("规则触碰放款件数量：" + str(fk_label_touch))
        print("规则触碰 M1+ 数量：")
        print(fk_label_touch_m1plus)


if __name__ == "__main__":
    header = '{"columns_one":"apply_id_no", "columns_two":"label", "columns_three":"due", "columns_four":"touch"}'
    cno = CountNeo4jOutput('\t', header, 'C:\\Users\\Dell\\Desktop\\output_mobile_counter_twentyone')
    cno.resolve_header()
    cno.load_data()
    cno.add_month()
    cno.modify_touch()
    cno.count_output()