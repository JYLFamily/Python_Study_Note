# coding:utf-8

import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
np.random.seed(9)


class AutoEncoderDemo(object):
    """ 先使用 DeepAutoEncoder 给 MNIST 降维到 32 维 , 后使用 TSNE 降维到 2 维并可视化

    # function
    ## function set 定义网络结构 , 包含 Encoder 与 Decoder
    ## goodness of function 定义损失函数 , 优化算法
    ## pick the best function 训练
    ## plotting 将 Encoder+Decoder 网络结构中 Encoder 的权重赋给新的 Encoder 使用新的 Encoder 降维

    """

    def __init__(self, *, batch_size, learning_rate, epochs):
        # data prepare
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        ## 降到 2 维
        self.__encoding_dim = 32

        # function set
        self.__input = Input(shape=(784, ))
        self.__encoded = None
        self.__decoded = None
        self.__encoder = None
        self.__decoder = None
        self.__autoencoder = None

        # goodness of function
        self.__learning_rate = learning_rate

        # pick the best function
        self.__batch_size = batch_size
        self.__epochs = epochs

        # plotting
        self.__net = None
        self.__viz = None

    def data_prepare(self):
        # load
        (self.__train, self.__train_label), (self.__test, self.__test_label) = mnist.load_data()

        # pre-processing
        self.__train = self.__train.reshape((self.__train.shape[0], -1)).astype("float32") / 255 - 0.5
        self.__test = self.__test.reshape((self.__test.shape[0], -1)).astype("float32") / 255 - 0.5

    def function_set(self):
        # encoder
        ## 输出是 128 输入是 784
        self.__encoded = Dense(units=128, activation="relu")(self.__input)
        ## 输出是 64 输入是上一个 self.__encoder  的输出
        self.__encoded = Dense(units=64, activation="relu")(self.__encoded)
        self.__encoded = Dense(units=10, activation="relu")(self.__encoded)
        self.__encoded = Dense(units=self.__encoding_dim)(self.__encoded)
        self.__encoder = Model(inputs=self.__input, outputs=self.__encoded)

        # decoder
        self.__decoded = Dense(units=10, activation="relu")(self.__encoded)
        self.__decoded = Dense(units=64, activation="relu")(self.__decoded)
        self.__decoded = Dense(units=128, activation="relu")(self.__decoded)
        self.__decoded = Dense(units=784, activation="tanh")(self.__decoded)
        self.__decoder = Model(inputs=self.__input, outputs=self.__decoded)

        # construct the autoencoder model
        self.__autoencoder = Model(inputs=self.__input, outputs=self.__decoded)

    def goodness_of_function(self):
        self.__autoencoder.compile(
            optimizer="adam",
            # 注意这里的 loss function
            loss="mse"
        )

    def pick_the_best_function(self):
        self.__autoencoder.fit(
            self.__train,
            self.__train,
            batch_size=self.__batch_size,
            epochs=self.__epochs,
            shuffle=True,
        )

    def plotting(self):
        # print(self.__autoencoder.layers[1].get_weights()[0].shape)
        # print(self.__autoencoder.layers[1].get_weights()[1].shape)

        # 使用 set_weights 一直失败
        self.__net = Sequential([
            Dense(units=128, input_dim=784,
                  activation="relu",
                  weights=self.__autoencoder.layers[1].get_weights()),
            Dense(units=64,
                  activation="relu",
                  weights=self.__autoencoder.layers[2].get_weights()),
            Dense(units=10,
                  activation="relu",
                  weights=self.__autoencoder.layers[3].get_weights()),
            Dense(units=self.__encoding_dim,
                  weights=self.__autoencoder.layers[4].get_weights())
        ])

        self.__viz = self.__net.predict(self.__train)
        self.__viz = TSNE(n_components=2).fit_transform(self.__viz)
        plt.scatter(self.__viz[:, 0], self.__viz[:, 1], c=self.__train_label)
        plt.show()

        self.__viz = self.__net.predict(self.__test)
        self.__viz = TSNE(n_components=2).fit_transform(self.__viz)
        plt.scatter(self.__viz[:, 0], self.__viz[:, 1], c=self.__test_label)
        plt.show()


if __name__ == "__main__":
    aed = AutoEncoderDemo(batch_size=64, learning_rate=0.1, epochs=5)
    aed.data_prepare()
    aed.function_set()
    aed.goodness_of_function()
    aed.pick_the_best_function()
    aed.plotting()




