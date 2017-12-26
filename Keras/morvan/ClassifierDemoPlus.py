# coding:utf-8

import numpy as np
import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


class ClassifierDemoPlus(object):

    def __init__(self):
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__model = None
        self.__gridsearch = None
        self.__params = {"batch_size": [10, 50, 100]}

    def __create_model(self):
        self.__model = Sequential([
            # 添加 "层" Dense(output_dim, input_dim)
            Dense(32, input_dim=784),
            # 添加 "激活函数"
            Activation("relu"),
            Dense(32),
            Activation("relu"),
            Dense(10),
            Activation("softmax")
        ])

        self.__model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"])

        return self.__model

    def train_test_split(self):
        # 必须要这么写与 sklearn train_test_split() 不同
        (self.__train, self.__train_label), (self.__test, self.__test_label) = mnist.load_data()

        # 三维数组展开变为二维数组
        # / float(255) 使用最大最小归一化 (min 0 max 255)
        self.__train = self.__train.reshape((self.__train.shape[0], -1)) / float(255)
        self.__test = self.__test.reshape((self.__test.shape[0], -1)) / float(255)

        # 多分类问题 y 需要 OneHotEncoder
        self.__train_label = to_categorical(self.__train_label, num_classes=10)
        self.__test_label = to_categorical(self.__test_label, num_classes=10)

    def train_model(self):
        self.__gridsearch = GridSearchCV(KerasClassifier(build_fn=self.__create_model), self.__params)
        self.__gridsearch.fit(self.__train, self.__train_label)
        print(self.__gridsearch.best_params_)


if __name__ == "__main__":
    cdp = ClassifierDemoPlus()
    cdp.train_test_split()
    cdp.train_model()