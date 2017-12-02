# coding:utf-8

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


class CnnDemo(object):

    def __init__(self):
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__model = Sequential()

    def load_data(self):
        (self.__train, self.__train_label), (self.__test, self.__test_label) = mnist.load_data()

        # (-1, 1, 28, 28) (batch, channels, height, width)
        self.__train = self.__train.reshape((-1, 1, 28, 28))
        self.__test = self.__test.reshape((-1, 1, 28, 28))
        self.__train_label = np_utils.to_categorical(self.__train_label, num_classes=10)
        self.__test_label = np_utils.to_categorical(self.__test_label, num_classes=10)

        print(self.__train.shape)
        print(self.__test.shape)
        print(self.__train_label.shape)
        print(self.__test_label.shape)

    def build_neural_network(self):
        # 通过 Convolution2D 扫描后得到一张图片的 32 张图片/特征 每个图片/特征都是与原图片大小相等的
        self.__model.add(Convolution2D(
            # 滤波器的个数 , 使用 32 个滤波器扫描图片 , 相当于获得 32 个特征
            filters=32,
            # 滤波器的大小 , width 与 height 分别是 5
            kernel_size=(5, 5),
            # 我理解为步长 , 滤波器移动的步长
            strides=(2, 2),
            # padding method
            padding="same",
            # 选择输入格式 batch 样本数
            # "channels_first"  (batch, height, width, channels)
            # "channels_last"   (batch, channels, height, width)
            data_format="channels_first",
            # Sequential 对象第一层必须有 input_shape or batch_input_shape
            input_shape=(1, 28, 28),
            #
            # batch_input_shape=()
        ))
        self.__model.add(Activation("relu"))
        # Pooling 下采样的过程 , 我理解就是缩减上面的 32 张图片/特征每个的大小
        self.__model.add(MaxPooling2D(
            # (height, width) 变为原先的一半
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
            data_format="channels_first"
        ))
        self.__model.add(Convolution2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            data_format="channels_first"))
        self.__model.add(Activation("relu"))
        self.__model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
            data_format="channels_first"
        ))
        self.__model.add(Flatten())
        # (64, 7, 7) 怎么变成了 1024 一张图片通过学习得到 64 个图片/特征 每个的大小都是 (7, 7)
        # 因为不是第一层 , 所以 1024 是该层输出的数目 ?
        self.__model.add(Dense(1024, name="Dense_1"))
        self.__model.add(Activation("relu"))
        self.__model.add(Dense(10, name="Dense_2"))
        self.__model.add(Activation("softmax"))

    def choose_loss_optimizing(self):
        self.__model.compile(optimizer=Adam(),
                             loss="categorical_crossentropy",
                             metrics=["accuracy"])

    def model_train(self):
        self.__model.fit(self.__train, self.__train_label, epochs=1, batch_size=32)

    def model_evaluate(self):
        loss , accuracy = self.__model.evaluate(self.__test, self.__test_label)

        print("test loss: ", loss)
        print("test accuracy: ", accuracy)

    def model_output(self):
        self.__model.get_layer("Dense_1").output


if __name__ == "__main__":
    cd = CnnDemo()
    cd.load_data()
    cd.build_neural_network()
    cd.choose_loss_optimizing()
    cd.model_train()
    cd.model_evaluate()
    cd.model_output()