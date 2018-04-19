# coding:utf-8

import keras
import pandas as pd
from keras import Sequential
from keras.layers import GRU
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


class YApplyTimeSeries(object):
    def __init__(self):
        # data prepare
        self.__df = None
        self.__train_feature_label, self.__test_feature_label = None, None
        self.__train_feature, self.__train_label = None, None
        self.__test_feature, self.__test_label = None, None
        self.__mms = None

        # function set
        self.__net = None

        # optimizer function

        # pick the best function

    def data_prepare(self):
        self.__df = pd.read_csv("C:\\Users\\Dell\\Desktop\\time_series.csv", encoding="utf-16")
        self.__df = self.__df.dropna()
        self.__train_feature_label = self.__df.loc[(self.__df["is_oot"] == 0), :]
        self.__test_feature_label = self.__df.loc[(self.__df["is_oot"] == 1), :]
        self.__train_feature_label = self.__train_feature_label.drop(["id_no", "is_oot"], axis=1)
        self.__test_feature_label = self.__test_feature_label.drop(["id_no", "is_oot"], axis=1)

        self.__train_feature = self.__train_feature_label[[i for i in self.__train_feature_label.columns if i != "is_overdue"]].values
        self.__train_label = self.__train_feature_label["is_overdue"].values
        self.__test_feature = self.__test_feature_label[[i for i in self.__test_feature_label.columns if i != "is_overdue"]].values
        self.__test_label = self.__test_feature_label["is_overdue"].values
        # 标准化
        self.__mms = MinMaxScaler()
        self.__mms.fit(self.__train_feature)
        self.__train_feature = self.__mms.transform(self.__train_feature)
        self.__test_feature = self.__mms.transform(self.__test_feature)
        # reshape samples × input_length × input_dim
        self.__train_feature = self.__train_feature.reshape((-1, 5, 3))
        self.__test_feature = self.__test_feature.reshape((-1, 5, 3))

    def function_set(self):
        self.__net = Sequential()
        self.__net.add(GRU(
            units=5,
            input_length=5,
            input_dim=3
        ))
        self.__net.add(Dense(
            units=1,
            activation="sigmoid"
        ))

    def optimizer_function(self):
        self.__net.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )

    def pick_the_best_function(self):
        self.__net.fit(self.__train_feature, self.__train_label, epochs=2, batch_size=256)
        print(roc_auc_score(self.__test_label, self.__net.predict_proba(self.__test_feature)))


if __name__ == "__main__":
    yats = YApplyTimeSeries()
    yats.data_prepare()
    yats.function_set()
    yats.optimizer_function()
    yats.pick_the_best_function()
