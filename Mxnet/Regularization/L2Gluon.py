# coding:utf-8

import random
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class Gluon(object):

    def __init__(self, *, num_train, num_test, num_inputs, learning_rate, epochs, weight_decay):
        # data prepare
        self.__num_train = num_train
        self.__num_test = num_test
        self.__num_inputs = num_inputs
        self.__true_w = nd.ones(shape=(num_inputs, 1)) * 0.01
        self.__true_b = 0.05
        self.__w = nd.random_normal(shape=(num_inputs, 1), scale=1)
        self.__b = nd.random_normal(shape=(1, 1), scale=1)
        self.__params = [self.__w, self.__b]

        self.__X = None
        self.__X_train, self.__X_test = None, None
        self.__y = None
        self.__y_train, self.__y_test = None, None

        # function set
        self.__net = None

        # goodness of function loss function
        self.__square_loss = None

        # goodness of function optimizer data
        self.__batch_size = 1

        # goodness of function optimizer function
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__trainer = None

        # pick the best function
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_yhat = None

    def data_prepare(self):
        self.__X = nd.random_normal(shape=(self.__num_train + self.__num_test, self.__num_inputs))
        self.__y = nd.dot(self.__X, self.__true_w)
        self.__y += .01 * nd.random_normal(shape=self.__y.shape)

        self.__X_train, self.__X_test = self.__X[:self.__num_train, :], self.__X[self.__num_train:, :]
        self.__y_train, self.__y_test = self.__y[:self.__num_train], self.__y[self.__num_train:]

    def function_set(self):
        self.__net = gluon.nn.Sequential()
        with self.__net.name_scope():
            self.__net.add(gluon.nn.Dense(1))
        self.__net.initialize()

    def goodness_of_function_loss_function(self):
        self.__square_loss = gluon.loss.L2Loss()

    def train_iter(self):
        idx = list(range(self.__num_train))
        random.shuffle(idx)
        for i in range(0, self.__num_train, self.__batch_size):
            j = nd.array(idx[i:min(i + self.__batch_size, self.__num_train)])
            yield self.__X_train.take(j), self.__y_train.take(j)

    def goodness_of_function_optimizer_function(self):
        self.__trainer = gluon.Trainer(self.__net.collect_params(), "sgd", {
            "learning_rate": self.__learning_rate, "wd": self.__weight_decay})

    def train_model(self):
        for param in self.__params:
            param.attach_grad()

        for e in range(self.__epochs):
            mean_train_loss = 0
            for self.__batch_X, self.__batch_y in self.train_iter():
                with autograd.record():
                     self.__batch_yhat = self.__net(self.__batch_X)
                     train_loss = self.__square_loss(self.__batch_yhat, self.__batch_y)
                train_loss.backward()
                self.__trainer.step(self.__batch_size)

                mean_train_loss += nd.mean(train_loss).asscalar()
            print("Epoch %d, train average loss: %f" % (e, mean_train_loss / self.__num_train))


if __name__ == "__main__":
    g = Gluon(num_train=5, num_test=100, num_inputs=20, learning_rate=0.1, epochs=10, weight_decay=0)
    g.data_prepare()
    g.function_set()
    g.goodness_of_function_loss_function()
    g.goodness_of_function_optimizer_function()
    g.train_model()