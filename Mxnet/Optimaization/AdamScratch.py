# coding:utf-8

import mxnet as mx
from mxnet import autograd
from mxnet import ndarray as nd
from mxnet import gluon
import random

mx.random.seed(1)
random.seed(1)


class AdamScratch(object):

    def __init__(self, *, batch_size, learning_rate, epochs):
        # data prepare
        self.__num_inputs = 2
        self.__num_examples = 1000
        self.__true_w = nd.array([2, -3.4]).reshape((2, 1))
        self.__true_b = nd.array([4.2]).reshape((1, 1))
        self.__w = None
        self.__b = None
        self.__params = None
        self.__sqrs = None
        self.__vs = None
        self.__t = None

        self.__X = None
        self.__y = None
        self.__data_set = None

        # function set

        # goodness of function loss function

        # goodness of function optimizer data
        self.__batch_size = batch_size
        self.__train_data_iter = None

        # goodness of function optimizer function
        self.__eps_stable = 1e-8
        self.__beta1 = 0.9
        self.__beta2 = 0.999
        self.__learning_rate = learning_rate

        # pick the best function
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def data_prepare(self):
        self.__X = nd.random_normal(shape=(self.__num_examples, self.__num_inputs), scale=1)
        self.__y = nd.dot(self.__X, self.__true_w) + self.__true_b
        self.__y += 0.1 * nd.random_normal(scale=1, shape=self.__y.shape)
        self.__data_set = gluon.data.ArrayDataset(self.__X, self.__y)

        self.__w = nd.random_normal(shape=(2, 1), scale=0.01)
        self.__b = nd.random_normal(shape=(1, 1), scale=0.01)
        self.__params = [self.__w, self.__b]
        self.__sqrs = []
        self.__vs = []

    def function_set(self):
        return nd.dot(self.__batch_X, self.__w) + self.__b

    def goodness_of_function_loss_function(self):
        return (self.__batch_y_hat - self.__batch_y.reshape(self.__batch_y_hat.shape)) ** 2 / 2

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(self.__data_set, self.__batch_size, shuffle=True)

    def goodness_of_function_optimizer_function(self):
        for param, v, sqr in zip(self.__params, self.__vs, self.__sqrs):
            g = param.grad / self.__batch_size
            v[:] = self.__beta1 * v + (1 - self.__beta1) * g
            sqr[:] = self.__beta2 * sqr + (1 - self.__beta2) * nd.square(g)
            v_hat = v / (1 - self.__beta1** self.__t)
            sqr_hat = sqr / (1 - self.__beta2 ** self.__t)
            div = self.__learning_rate * v_hat / nd.sqrt(sqr_hat + self.__eps_stable)
            param[:] -= div


    def pick_the_best_function(self):
        for param in self.__params:
            param.attach_grad()
            self.__sqrs.append(param.zeros_like())
            self.__vs.append(param.zeros_like())

        for e in list(range(self.__epochs)):
            train_loss = 0.

            self.__t = 0
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    loss = self.goodness_of_function_loss_function()
                loss.backward()
                # 调用 Adam 算法前更新 t
                self.__t += 1
                self.goodness_of_function_optimizer_function()

                train_loss += nd.mean(loss).asscalar()
            print("Epoch %d. Learning Rate %f. Loss: %f." % (e, self.__learning_rate, train_loss))
        print(self.__w, sep=" ")
        print(self.__b)


if __name__ == "__main__":
    rms = AdamScratch(batch_size=10, learning_rate=0.03, epochs=10)
    rms.data_prepare()
    rms.goodness_of_function_optimizer_data()
    rms.pick_the_best_function()