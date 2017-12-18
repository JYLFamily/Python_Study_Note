# coding:utf-8

import random
import numpy as np
from mxnet import ndarray as nd
from mxnet import autograd
np.random.seed(9)


class Scratch(object):

    def __init__(self):
        # X shape (1000, 2)
        self.__num_inputs = 2
        self.__num_examples = 1000

        self.__true_w = np.array([2, -3.4]).reshape((2, 1))
        self.__true_b = 4.2

        self.__X = np.random.normal(size=(self.__num_examples, self.__num_inputs))
        self.__y = np.dot(self.__X, self.__true_w) + self.__true_b
        self.__y = self.__y + 0.01 * np.random.normal(size=(self.__num_examples, 1))
        # np.array -> nd.array
        self.__X = nd.array(self.__X)
        self.__y = nd.array(self.__y)

        self.__w = nd.array(np.random.normal(size=(2, 1)))
        self.__b = nd.array(np.random.normal(size=(1, 1)))
        self.__params = [self.__w, self.__b]

        self.__batch_size = 500
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

        self.__epochs = 5
        self.__learning_rate = 0.001

    def attach_gradient(self):
        for param in self.__params:
            param.attach_grad()

    def data_iter(self):
        idx = list(range(self.__num_examples))
        random.shuffle(idx)
        for i in range(0, self.__num_examples, self.__batch_size):
            j = nd.array(idx[i:min(i + self.__batch_size, self.__num_examples)])
            yield nd.take(self.__X, j).reshape((-1, 2)), nd.take(self.__y, j).reshape((-1, 1))

    def net(self):
        return nd.dot(self.__batch_X, self.__w) + self.__b

    def square_loss(self):
        return (self.__batch_y_hat - self.__batch_y) ** 2

    def stochastic_gradient_descent(self):
        for param in self.__params:
            param[:] = param - self.__learning_rate * param.grad

    def train_model(self):
        for e in range(self.__epochs):
            total_loss = 0
            for self.__batch_X, self.__batch_y in self.data_iter():
                with autograd.record():
                    self.__batch_y_hat = self.net()
                    loss = self.square_loss()
                loss.backward()
                self.stochastic_gradient_descent()
                total_loss += nd.sum(loss).asscalar()
            print("Epoch %d, average loss: %f" % (e, total_loss))


if __name__ == "__main__":
    s = Scratch()
    s.attach_gradient()
    s.train_model()