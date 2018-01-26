# coding:utf-8

import random
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
np.random.seed(9)


class RnnLstmDemo(object):

    def __init__(self, *, units, input_length, input_dim, output_dim):
        # function set
        self.__net = Sequential()
        self.__units = units
        self.__input_length = input_length
        self.__input_dim = input_dim
        self.__output_dim = output_dim

        # goodness of function

        # pick the best function

    def function_set(self):
        self.__net.add(LSTM(
            units=self.__units,
            input_length=self.__input_length,
            input_dim=self.__input_dim,
            return_sequences=False
        ))
        self.__net.add(Dense(
            units=self.__output_dim,
            activation="softmax"
        ))

        print(self.__net.summary())
        for layer in self.__net.layers:
            print(layer.input_shape, layer.output_shape)
            # print(layer.get_weights())

    def goodness_of_function(self):
        self.__net.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["acc"]
        )

    def pick_the_best_function(self):
        pass


if __name__ == "__main__":
    rld = RnnLstmDemo(
        units=4,
        input_length=1,
        input_dim=3,
        output_dim=3
    )
    rld.function_set()
