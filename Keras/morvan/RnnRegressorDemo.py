# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
np.random.seed(9)


class RnnRegressor(object):
    def __init__(self, *, input_path, look_back, batch_size, epochs):
        # data prepare
        self.__file = pd.read_csv(input_path, usecols=[1]).values
        self.__look_back = look_back
        self.__X, self.__y = [None for _ in range(2)]
        self.__train_scaler, self.__test_scaler = [None for _ in range(2)]
        self.__train, self.__train_label, self.__test, self.__test_label = [None for _ in range(4)]

        # function set
        self.__net = None

        # goodness of function
        self.__batch_size = batch_size

        # pick the best function
        self.__epochs = epochs

    def data_prepare(self):
        def create_X_y(file=self.__file, look_back=self.__look_back):
            X = []
            y = []
            for i in range(len(file) - look_back - 1):
                X.append(file[i:(i+look_back), 0])
                y.append(file[i+look_back, 0])
            return np.array(X).reshape((-1, look_back)), np.array(y).reshape((-1, 1))
        self.__X, self.__y = create_X_y()

        self.__train, self.__test, self.__train_label, self.__test_label = train_test_split(
            self.__X,
            self.__y,
            test_size=0.2,
            shuffle=False
        )

        self.__train_scaler = StandardScaler().fit(self.__train)
        self.__train = self.__train_scaler.transform(self.__train)
        self.__test = self.__train_scaler.transform(self.__test)
        self.__train = self.__train.reshape((-1, self.__look_back, 1))
        self.__test = self.__test.reshape((-1, self.__look_back, 1))

        self.__test_scaler = StandardScaler().fit(self.__train_label)
        self.__train_label = self.__test_scaler.transform(self.__train_label)

    def function_set(self):
        self.__net = Sequential([
            LSTM(units=4, input_length=self.__look_back, input_dim=1),
            Dense(1)
        ])

    def goodness_of_function(self):
        self.__net.compile(loss="mse",optimizer="adam")

    def pick_the_best_function(self):
        self.__net.fit(self.__train, self.__train_label, batch_size=self.__batch_size, epochs=self.__epochs, verbose=2)
        plt.plot(self.__test_scaler.inverse_transform(self.__net.predict(self.__test)))
        plt.plot(self.__test_label)
        plt.show()


if __name__ == "__main__":
    rr = RnnRegressor(
        input_path="international-airline-passengers.csv",
        look_back=3,
        batch_size=1,
        epochs=200
    )
    rr.data_prepare()
    rr.function_set()
    rr.goodness_of_function()
    rr.pick_the_best_function()