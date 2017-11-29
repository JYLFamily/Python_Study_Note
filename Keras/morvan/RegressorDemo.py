# coding:utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(1337)


class RegressorDemo(object):

    def __init__(self):
        self.__X = np.linspace(-1, 1, 200).reshape((200, 1))
        np.random.shuffle(self.__X)
        self.__y = 0.5 * self.__X + 2 + np.random.normal(0, 0.05, (200, 1))
        self.__train = self.__X[0:160]
        self.__train_label = self.__y[0:160]
        self.__test = self.__X[160:]
        self.__test_label = self.__y[160:]
        self.__model = Sequential()

    def build_neural_network(self):
        self.__model.add(Dense(output_dim=1, input_dim=1))

    def choose_loss_optimizing(self):
        self.__model.compile(loss="mse", optimizer="sgd")

    def model_train(self):
        for step in range(0, 301):
            cost = self.__model.train_on_batch(self.__train, self.__train_label)
            if step % 100 == 0:
                print("train cost: ", cost)

    def model_test(self):
        cost = self.__model.evaluate(self.__test, self.__test_label, batch_size=40)
        print("test cost:", cost)
        W, b = self.__model.layers[0].get_weights()
        print("Weights=", W, "\nbiases=", b)


if __name__ == "__main__":
    rd = RegressorDemo()
    rd.build_neural_network()
    rd.choose_loss_optimizing()
    rd.model_train()
    rd.model_test()