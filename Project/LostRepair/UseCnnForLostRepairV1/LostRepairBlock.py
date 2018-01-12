# coding:utf-8

from mxnet import nd
from mxnet.gluon import nn


class LostRepairBlock(nn.Block):
    def __init__(self, **kwargs):
        super(LostRepairBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.__d1 = nn.Dense(units=16, activation="relu")
            self.__c1 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1, activation="relu")
            self.__c2 = nn.Conv2D(64, kernel_size=3, strides=1, padding=1, activation="relu")
            self.__c3 = nn.Conv2D(1, kernel_size=1, strides=1, padding=1, activation="relu")
            self.__f1 = nn.MaxPool2D()
            self.__f2 = nn.Flatten()
            self.__d2 = nn.Dense(units=32, activation="relu")
            self.__d3 = nn.Dense(units=64, activation="relu")
            self.__d4 = nn.Dense(units=1)
            # self.__p1 = nn.AvgPool2D(pool_size=4)
            # self.__f1 = nn.Flatten()

    def forward(self, x):
        x = self.__d1(x)
        x = x.reshape((-1, 1, 4, 4))
        x = self.__c1(x)
        x = self.__c2(x)
        x = self.__c3(x)
        x = self.__f1(x)
        x = self.__f2(x)
        x = self.__d2(x)
        x = self.__d3(x)
        x = self.__d4(x)

        return x


if __name__ == "__main__":
    x = nd.arange(25).reshape((1, 1, 5, 5))
    lbk = LostRepairBlock()
    lbk.initialize()
    print(lbk(x).shape)
