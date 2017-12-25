# coding:utf-8

import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras import backend as K
np.random.seed(9)


class ClassifierDemo(object):

    def __init__(self):
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__model = None

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

    def build_neural_network(self):
        self.__model = Sequential([
            # 添加 "层" Dense(output_dim, input_dim)
            Dense(32, input_dim=784),
            # 添加 "激活函数"
            Activation("relu"),
            Dense(32),
            Activation("relu"),
            Dense(10),
            Activation("softmax"),
        ])

    def choose_loss_optimizing(self):
        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # loss 与 optimizer 是什么意思 ? 损失函数与优化方法 ? 那我觉得写反了
        # 刚才就是写反了
        self.__model.compile(loss="categorical_crossentropy",
                             optimizer=rmsprop,
                             metrics=["accuracy"])

    def model_train(self):
        print("Training ------------")
        # Another way to train the model 之前是 train_on_batch()
        self.__model.fit(self.__train, self.__train_label, epochs=2, batch_size=32)

    def model_test(self):
        print("\nTesting ------------")
        # Evaluate the model with the metrics we defined earlier 将 predict 与评估合并到了一起变为 evaluate
        loss, accuracy = self.__model.evaluate(self.__test, self.__test_label)

        print("test loss: ", loss)
        print("test accuracy: ", accuracy)

    def model_output(self):
        intermediate_tensor_function = K.function([self.__model.layers[0].input],
                                                  [self.__model.layers[3].output])
        intermediate_tensor = intermediate_tensor_function([self.__test])[0]
        print(type(intermediate_tensor_function))
        print(intermediate_tensor.shape)
        print(intermediate_tensor)


if __name__ == "__main__":
    cd = ClassifierDemo()
    cd.train_test_split()
    cd.build_neural_network()
    cd.choose_loss_optimizing()
    cd.model_train()
    # cd.model_test()
    cd.model_output()