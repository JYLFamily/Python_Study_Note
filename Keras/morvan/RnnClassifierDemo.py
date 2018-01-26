# coding:utf-8

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
np.random.seed(9)


class RnnClassifierDemo(object):

    def __init__(self, *, batch_size, learning_rate, epochs):
        # data prepare
        self.__train, self.__train_label, self.__test, self.__test_label = [None for _ in range(4)]
        self.__batch_size = batch_size
        #  照片每一行是一个 time_step 28×28 的图片共有 28 个 step
        self.__time_steps = 28
        self.__input_size = 28

        #  10 分类问题
        self.__output_size = 10
        #  Rnn 中有 50 个 "hidden unit"
        self.__cell_size = 50

        # function set
        self.__net = None

        # goodness of function
        self.__learning_rate = learning_rate

        # pick the best function
        self.__epochs = epochs

    def data_prepare(self):
        # load
        (self.__train, self.__train_label), (self.__test, self.__test_label) = mnist.load_data()

        # pre-processing
        self.__train = self.__train.reshape((-1, 28, 28)).astype("float32") / 255
        self.__test = self.__test.reshape((-1, 28, 28)).astype("float32") / 255
        self.__train_label = np_utils.to_categorical(self.__train_label, num_classes=self.__output_size)
        self.__test_label = np_utils.to_categorical(self.__test_label, num_classes=self.__output_size)

    def function_set(self):
        self.__net = Sequential()
        self.__net.add(SimpleRNN(
            units=self.__cell_size,
            input_dim=self.__input_size,
            input_length=self.__time_steps
        ))
        self.__net.add(Dense(
            units=self.__output_size
        ))
        self.__net.add(Activation(
            "softmax"
        ))

    def goodness_of_function(self):
        self.__net.compile(
            optimizer=Adam(lr=self.__learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def pick_the_best_function(self):
        self.__net.fit(
            self.__train,
            self.__train_label,
            batch_size=self.__batch_size,
            epochs=self.__epochs,
            shuffle=True,
        )


if __name__ == "__main__":
    rcd = RnnClassifierDemo(batch_size=64, learning_rate=0.001, epochs=5)
    rcd.data_prepare()
    rcd.function_set()
    rcd.goodness_of_function()
    rcd.pick_the_best_function()