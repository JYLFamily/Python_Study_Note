# coding:utf-8

import json
import pandas as pd


def binning(x, bins):
    for i in range(0, 5):
        if bins[i] < x <= bins[i+1]:
            return bins[i+1]


class Demo(object):
    def __init__(self):
        self.__json_dict = None
        self.__json_dict_gsm = None
        self.__json_dict_sms = None
        self.__json_dict_gsm_df = None
        self.__json_dict_sms_df = None

        self.__merge_feature = None

    def load_json(self):
        with open("C:\\Users\\Dell\\Desktop\\demo.json", mode="r", encoding="utf-8") as f:
            self.__json_dict = json.load(f)

    def gsm_feature(self):
        self.__json_dict_gsm = self.__json_dict["result"]["mobileDetailGsm"]
        self.__json_dict_gsm_df = pd.DataFrame()

        for i in self.__json_dict_gsm.keys():
            for j in self.__json_dict_gsm[i]:
                self.__json_dict_gsm_df = self.__json_dict_gsm_df.append(pd.DataFrame.from_dict(j, orient="index").T)
        # 通话开始时间, 通话时长(秒), 通话类型, 对方电话是否正常, 对方电话是否是催收电话
        self.__json_dict_gsm_df = self.__json_dict_gsm_df[["startTime", "communicationDurationNum", "communicationWay", "counterpartPhoneIsNormal", "isurge"]]

        self.__json_dict_gsm_df["startTime"] = pd.to_datetime(self.__json_dict_gsm_df["startTime"])
        self.__json_dict_gsm_df["startTime"] = self.__json_dict_gsm_df["startTime"].apply(
            lambda x: binning(x, pd.date_range(end="2018-04-17", periods=6, freq="30D")))
        self.__json_dict_gsm_df = self.__json_dict_gsm_df.dropna()

        # 通话时点标志位
        self.__json_dict_gsm_df["startTime_is_before_dawn"] = self.__json_dict_gsm_df["startTime"].apply(lambda x: 1 if x.hour <= 6 else 0)
        self.__json_dict_gsm_df["startTime_is_dawn"] = self.__json_dict_gsm_df["startTime"].apply(lambda x: 1 if x.hour > 6 and x.hour <= 8 else 0)
        self.__json_dict_gsm_df["startTime_is_morning"] = self.__json_dict_gsm_df["startTime"].apply(lambda x: 1 if x.hour > 8 and x.hour <= 12 else 0)
        self.__json_dict_gsm_df["startTime_is_afternoon"] = self.__json_dict_gsm_df["startTime"].apply(lambda x: 1 if x.hour > 12 and x.hour <= 19 else 0)
        self.__json_dict_gsm_df["startTime_is_evening"] = self.__json_dict_gsm_df["startTime"].apply(lambda x: 1 if x.hour > 19 and x.hour <= 24 else 0)
        # 通话方式标志位
        self.__json_dict_gsm_df["communicationWay_calling"] = self.__json_dict_gsm_df["communicationWay"].apply(lambda x: 1 if x == "主叫" else 0)
        self.__json_dict_gsm_df["communicationWay_called"] = self.__json_dict_gsm_df["communicationWay"].apply(lambda x: 1 if x == "被叫" else 0)

        for i in ["calling", "called"]:
            for j in ["is_before_dawn", "is_dawn", "is_morning", "is_afternoon", "is_afternoon", "is_evening"]:
                self.__json_dict_gsm_df[i + "_" + j + "_count"] = self.__json_dict_gsm_df.apply(
                    lambda x: 1 if x["communicationWay_"+i] == 1 and x["startTime_"+j] == 1 else 0, axis=1)
                self.__json_dict_gsm_df[i + "_" + j + "_num"] = self.__json_dict_gsm_df.apply(
                    lambda x: x["communicationDurationNum"] if x["communicationWay_"+i] == 1 and x["startTime_"+j] == 1 else 0, axis=1)

        self.__json_dict_gsm_df = self.__json_dict_gsm_df.drop(["communicationDurationNum", "communicationWay",
                                                                "startTime_is_before_dawn", "startTime_is_dawn",
                                                                "startTime_is_morning", "startTime_is_afternoon",
                                                                "startTime_is_evening",
                                                                "communicationWay_calling", "communicationWay_called"], axis=1)
        self.__json_dict_gsm_df["startTime"] = self.__json_dict_gsm_df["startTime"].apply(lambda x: x.strftime("%Y-%m-%d"))
        self.__json_dict_gsm_df = self.__json_dict_gsm_df.groupby(["startTime"])[[i for i in self.__json_dict_gsm_df.columns if i != "startTime"]].sum()
        self.__json_dict_gsm_df = self.__json_dict_gsm_df.reset_index(drop=False)

    def sms_feature(self):
        self.__json_dict_sms = self.__json_dict["result"]["mobileDetailSms"]
        self.__json_dict_sms_df = pd.DataFrame()

        for i in self.__json_dict_sms.keys():
            for j in self.__json_dict_sms[i]:
                self.__json_dict_sms_df = self.__json_dict_sms_df.append(pd.DataFrame.from_dict(j, orient="index").T)

        self.__json_dict_sms_df = self.__json_dict_sms_df[["startTime", "communicationWay"]]
        self.__json_dict_sms_df["startTime"] = pd.to_datetime(self.__json_dict_sms_df["startTime"])
        self.__json_dict_sms_df["startTime"] = self.__json_dict_sms_df["startTime"].apply(
            lambda x: binning(x, pd.date_range(end="2018-04-17", periods=6, freq="30D")))
        self.__json_dict_sms_df = self.__json_dict_sms_df.dropna()

        self.__json_dict_sms_df["send_count"] = self.__json_dict_sms_df["communicationWay"].apply(lambda x: 1 if x == "发" else 0)
        self.__json_dict_sms_df["receive_count"] = self.__json_dict_sms_df["communicationWay"].apply(lambda x: 1 if x == "收" else 0)

        self.__json_dict_sms_df = self.__json_dict_sms_df.drop(["communicationWay"], axis=1)
        self.__json_dict_sms_df["startTime"] = self.__json_dict_sms_df["startTime"].apply(lambda x: x.strftime("%Y-%m-%d"))
        self.__json_dict_sms_df = self.__json_dict_sms_df.groupby(["startTime"])[[i for i in self.__json_dict_sms_df.columns if i != "startTime"]].sum()
        self.__json_dict_sms_df = self.__json_dict_sms_df.reset_index(drop=False)

    def merge_feature(self):
        self.__merge_feature = pd.Series(pd.date_range(end="2018-04-17", periods=5, freq="30D")).apply(lambda x: x.strftime("%Y-%m-%d")).to_frame("time_series")
        self.__merge_feature = self.__merge_feature.merge(self.__json_dict_gsm_df, left_on=["time_series"], right_on=["startTime"], how="left")
        self.__merge_feature = self.__merge_feature.drop(["startTime"], axis=1)
        self.__merge_feature = self.__merge_feature.merge(self.__json_dict_sms_df, left_on=["time_series"], right_on=["startTime"], how="left")
        self.__merge_feature = self.__merge_feature.drop(["startTime"], axis=1)
        self.__merge_feature = self.__merge_feature.fillna(0)


if __name__ == "__main__":
    d = Demo()
    d.load_json()
    d.gsm_feature()
    d.sms_feature()
    d.merge_feature()



