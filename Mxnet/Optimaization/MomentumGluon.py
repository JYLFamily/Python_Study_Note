# coding:utf-8

import mxnet as mx
from mxnet import autograd
from mxnet import ndarray as nd
from mxnet import gluon
import random

mx.random.seed(1)
random.seed(1)


class MomentumGluon(object):
    def __init__(self, *, batch_size, learning_rate, momentum, epochs):
        # data prepare
        self.__num_inputs = 2
        self.__num_examples = 1000
        self.__true_w = nd.array([2, -3.4]).reshape((2, 1))
        self.__true_b = nd.array([4.2]).reshape((1, 1))
        self.__w = None
        self.__b = None
        self.__params = None

        self.__X = None
        self.__y = None
        self.__data_set = None

        # function set
        self.__net = None

        # goodness of function loss function
        self.__loss = None

        # goodness of function optimizer data
        self.__batch_size = batch_size
        self.__train_data_iter = None

        # goodness of function optimizer function
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__trainer = None

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

    def function_set(self):
        self.__net = gluon.nn.HybridSequential()
        with self.__net.name_scope():
            self.__net.add(gluon.nn.Dense(1))
            self.__net.initialize()
            self.__net.hybridize()

    def goodness_of_function_loss_function(self):
        self.__loss = gluon.loss.L2Loss()

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(self.__data_set, self.__batch_size, shuffle=True)

    def goodness_of_function_optimizer_function(self):
        self.__trainer = gluon.Trainer(
            self.__net.collect_params(),
            "sgd",
            {"learning_rate": self.__learning_rate,
             "momentum": self.__momentum}
        )

    def pick_the_best_function(self):
        for e in list(range(self.__epochs)):
            train_loss = 0.
            self.__trainer.set_learning_rate(self.__trainer.learning_rate * 0.1) if e > 2 else self.__trainer
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                with autograd.record():
                    self.__batch_y_hat = self.__net(self.__batch_X)
                    loss = self.__loss(self.__batch_y_hat, self.__batch_y)
                loss.backward()
                self.__trainer.step(self.__batch_size)

                train_loss += nd.mean(loss).asscalar()
            print("Epoch %d. Learning Rate %f. Loss: %f." % (e, self.__trainer.learning_rate, train_loss))
        print(self.__net[0].weight.data())
        print(self.__net[0].bias.data())


if __name__ == "__main__":
    mg = MomentumGluon(batch_size=10, learning_rate=0.2, momentum=0.9, epochs=5)
    mg.data_prepare()
    mg.function_set()
    mg.goodness_of_function_loss_function()
    mg.goodness_of_function_optimizer_data()
    mg.goodness_of_function_optimizer_function()
    mg.pick_the_best_function()
