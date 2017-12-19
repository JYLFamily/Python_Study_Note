# coding:utf-8

import random
from mxnet import ndarray as nd
from mxnet import autograd


class Scratch(object):

    def __init__(self):
        # 构造测试数据
        self.__num_inputs = None
        self.__num_examples = None
        self.__true_w = None
        self.__true_b = None
        self.__X = None
        self.__y = None

        # function set 待估计参数
        self.__w = None
        self.__b = None
        self.__params = None

        # goodness of function loss function

        # goodness of function optimizer data
        self.__batch_size = None

        # goodness of function optimizer function
        self.__learning_rate = None

        # pick the best function 模型训练
        self.__epochs = None
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def data_prepare(self):
        self.__num_inputs = 2
        self.__num_examples = 1000
        self.__true_w = nd.array([2, -3.4]).reshape((2, 1))
        self.__true_b = nd.array([4.2]).reshape((1, 1))
        self.__X = nd.random_normal(shape=(self.__num_examples, self.__num_inputs))
        self.__y = nd.dot(self.__X, self.__true_w) + self.__true_b
        self.__y += 0.001 * nd.random_normal(shape=self.__y.shape)

        self.__w = nd.random_normal(shape=(2, 1))
        self.__b = nd.random_normal(shape=(1, 1))
        self.__params = [self.__w, self.__b]

    def function_set(self):
        return nd.dot(self.__batch_X, self.__w) + self.__b

    def goodness_of_function_loss_function(self):
        return (self.__batch_y - self.__batch_y_hat) ** 2

    def goodness_of_function_optimizer_data(self):
        self.__batch_size = 100
        idx = list(range(self.__num_examples))
        random.shuffle(idx)
        for i in range(0, self.__num_examples, self.__batch_size):
            j = nd.array(idx[i:min(i + self.__batch_size, self.__num_examples)])
            yield nd.take(self.__X, j).reshape((-1, 2)), nd.take(self.__y, j).reshape((-1, 1))

    def goodness_of_function_optimizer_function(self):
        self.__learning_rate = 0.001
        for param in self.__params:
            param[:] = param - self.__learning_rate * param.grad

    def train_model(self):
        for param in self.__params:
            param.attach_grad()

        self.__epochs = 5
        for e in range(self.__epochs):
            total_loss = 0
            for self.__batch_X, self.__batch_y in self.goodness_of_function_optimizer_data():
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    loss = self.goodness_of_function_loss_function()
                loss.backward()
                self.goodness_of_function_optimizer_function()
                total_loss += nd.sum(loss).asscalar()
            print("Epoch %d, average loss: %f" % (e, total_loss))


if __name__ == "__main__":
    s = Scratch()
    s.data_prepare()
    s.train_model()