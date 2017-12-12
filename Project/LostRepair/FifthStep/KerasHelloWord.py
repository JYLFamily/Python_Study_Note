# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score


class KerasHelloWorld(object):

    def __init__(self, input_path, test_size, random_state):
        self.__X = pd.read_csv(input_path, usecols=list(range(1, 53))).values
        self.__y = pd.read_csv(input_path, usecols=[0]).values
        self.__test_size = test_size
        self.__random_state = random_state
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__model = None

    def train_test_split(self):
        self.__train, self.__test, self.__train_label, self.__test_label = (train_test_split(
            self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))
        #
        scaler = StandardScaler()
        scaler.fit(self.__train)
        self.__train = scaler.transform(self.__train)
        self.__test = scaler.transform(self.__test)

    def define_fun_set(self):
        self.__model = Sequential()
        self.__model.add(Dense(input_dim=52, units=600, activation="relu"))
        # self.__model.add(Dropout(0.5))
        self.__model.add(Dense(units=600, activation="relu"))
        # self.__model.add(Dropout(0.5))
        self.__model.add(Dense(units=600, activation="relu"))
        # self.__model.add(Dropout(0.5))
        self.__model.add(Dense(units=600, activation="relu"))
        # self.__model.add(Dropout(0.5))
        self.__model.add(Dense(units=1, activation="sigmoid"))

    def goodness_of_fun(self):
        self.__model.compile(loss="binary_crossentropy",
                             optimizer="adam",
                             metrics=["accuracy"])

    def train(self):
        self.__model.fit(self.__train, self.__train_label, batch_size=10, epochs=2)

    def evaluate_training_set(self):
        loss, accuracy = self.__model.evaluate(self.__train, self.__train_label, batch_size=10)
        print("train loss: ", loss)
        print("train accuracy: ", accuracy)

    def evaluate_testing_set(self):
        loss, accuracy = self.__model.evaluate(self.__test, self.__test_label, batch_size=10)
        print("test loss: ", loss)
        print("test accuracy: ", accuracy)

    def predict(self):
        print(roc_auc_score(self.__test_label, self.__model.predict_proba(self.__test)))


if __name__ == "__main__":
    khw = KerasHelloWorld(input_path="D:\\Project\\LostRepair\\yunyingshang\\all.csv",
                          test_size=0.2,
                          random_state=9)
    khw.train_test_split()
    khw.define_fun_set()
    khw.goodness_of_fun()
    khw.train()
    # khw.evaluate_training_set()
    # khw.evaluate_testing_set()
    khw.predict()