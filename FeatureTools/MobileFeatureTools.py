# coding:utf-8

import os
import re
import numpy as np
import pandas as pd
import featuretools as ft


class MobileFeatureTools(object):
    def __init__(self, *, path):
        # set df
        self.__path = path
        self.__basic = None
        self.__cutoff = None
        self.__trans_sms = None
        self.__trans_voice = None

        # set es
        self.__es = None

    def set_df(self):
        self.__basic = pd.read_csv(os.path.join(self.__path, "basic.csv"), header=None, na_values=["Null"])
        self.__cutoff = pd.read_csv(os.path.join(self.__path, "cutoff.csv"), header=None, na_values=["Null"])
        self.__trans_sms = pd.read_csv(os.path.join(self.__path, "trans_sms.csv"), header=None, na_values=["Null"])
        self.__trans_voice = pd.read_csv(os.path.join(self.__path, "trans_voice.csv"), header=None, na_values=["Null"])

        self.__basic.columns = ["user_id", "create_time", "open_date", "account_balance", "integral"]
        self.__cutoff.columns = ["user_id", "create_time", "is_oot", "is_overdue"]
        self.__trans_sms.columns = ["user_id", "communication_way", "start_time"]
        self.__trans_voice.columns = ["user_id", "communication_fee", "communication_way", "start_time"]

    def clean_df(self):
        self.__basic["create_time"] = pd.to_datetime(self.__basic["create_time"])
        self.__basic["open_date"] = pd.to_datetime(self.__basic["open_date"])
        # (self.__basic["create_time"] - self.__basic["open_date"]) 返回 Series
        self.__basic["open_date"] = (self.__basic["create_time"] - self.__basic["open_date"]).apply(lambda x: x.days)
        self.__basic = self.__basic.drop(["create_time"], axis=1)
        # basic 表去重
        self.__basic = self.__basic.loc[np.logical_not(self.__basic["user_id"].duplicated()), :]

        # 字段 type 不正确 , 过滤掉
        self.__trans_sms = self.__trans_sms.loc[(self.__trans_sms["start_time"].apply(lambda x: True if type(x) == str else False)), :]
        # str 格式不正确
        self.__trans_sms = self.__trans_sms.loc[(self.__trans_sms["start_time"].apply(lambda x: True if re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", x) else False)), :]
        self.__trans_sms["start_time"] = pd.to_datetime(self.__trans_sms["start_time"])
        # 接收 1 发送 0
        self.__trans_sms["communication_way"] = self.__trans_sms["communication_way"].apply(lambda x: 1 if x == "接收" else 0)
        self.__trans_sms["trans_sms_id"] = list(range(self.__trans_sms.shape[0]))

        # 字段 type 不正确 , 过滤掉
        self.__trans_voice = self.__trans_voice.loc[(self.__trans_voice["start_time"].apply(lambda x: True if type(x) == str else False)), :]
        # str 格式不正确
        self.__trans_voice = self.__trans_voice.loc[(self.__trans_voice["start_time"].apply(lambda x: True if re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", x) else False)), :]
        self.__trans_voice["start_time"] = pd.to_datetime(self.__trans_voice["start_time"])
        # 被叫 1 主叫 0
        self.__trans_voice["communication_way"] = self.__trans_voice["communication_way"].apply(lambda x: 1 if x == "被叫" else 0)
        self.__trans_voice["trans_voice_id"] = list(range(self.__trans_voice.shape[0]))

    def set_es(self):
        self.__es = ft.EntitySet(id="basic")
        self.__es = self.__es.entity_from_dataframe(
            entity_id="basic",
            dataframe=self.__basic,
            index="user_id"
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="trans_sms",
            dataframe=self.__trans_sms,
            index="trans_sms_id",
            time_index="start_time"
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="trans_voice",
            dataframe=self.__trans_voice,
            index="trans_voice_id",
            time_index="start_time"
        )
        # self.__es = self.__es.add_relationship(
        #     ft.Relationship(
        #         self.__es[""basic""]
        #     )
        # )


if __name__ == "__main__":
    mft = MobileFeatureTools(path="C:\\Users\\Dell\\Desktop\\week\\FC\\score_card\\yunyingshang\\2018-04-28\\data")
    mft.set_df()
    mft.clean_df()
    mft.set_es()