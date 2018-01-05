# coding:utf-8

import random
from mxnet import ndarray as nd
from mxnet import autograd


class L2Scratch(object):

    def __init__(self, *, num_train, num_test, num_inputs, learning_rate, epochs, lamda):
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

        # goodness of function loss function
        self.__lamda = lamda

        # goodness of function optimizer data
        self.__batch_size = 1

        # goodness of function optimizer function
        self.__learning_rate = learning_rate

        # pick the best function
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def data_prepare(self):
        self.__X = nd.random_normal(shape=(self.__num_train + self.__num_test, self.__num_inputs))
        self.__y = nd.dot(self.__X, self.__true_w)
        self.__y += .01 * nd.random_normal(shape=self.__y.shape)

        self.__X_train, self.__X_test = self.__X[:self.__num_train, :], self.__X[self.__num_train:, :]
        self.__y_train, self.__y_test = self.__y[:self.__num_train], self.__y[self.__num_train:]

    # 模型是 Xw + b
    def function_set(self):
        return nd.dot(self.__batch_X, self.__w) + self.__b

    # loss是 Xw + b + L2
    def goodness_of_function_loss_function(self):
        def square_loss(yhat, y):
            return (yhat - y.reshape(yhat.shape)) ** 2 / 2

        def l2_penalty():
            return ((self.__w**2).sum() + self.__b**2) / 2

        return square_loss(self.__batch_y_hat, self.__batch_y) + self.__lamda * l2_penalty()

    def train_iter(self):
        idx = list(range(self.__num_train))
        random.shuffle(idx)
        for i in range(0, self.__num_train, self.__batch_size):
            j = nd.array(idx[i:min(i + self.__batch_size, self.__num_train)])
            yield self.__X_train.take(j), self.__y_train.take(j)

    def goodness_of_function_optimizer_function(self):
        for param in self.__params:
            param[:] = param - self.__learning_rate / self.__batch_size * param.grad

    def train_model(self):
        for param in self.__params:
            param.attach_grad()

        for e in range(self.__epochs):
            mean_train_loss = 0
            mean_test_loss = 0
            for self.__batch_X, self.__batch_y in self.train_iter():
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    train_loss = self.goodness_of_function_loss_function()
                train_loss.backward()
                self.goodness_of_function_optimizer_function()

                mean_train_loss += nd.mean(train_loss).asscalar()

                test_y_hat = nd.dot(self.__X_test, self.__w) + self.__b
                test_loss = ((test_y_hat - self.__y_test) ** 2 / 2 +
                              self.__lamda * ((self.__w**2).sum() + self.__b**2) / 2)
                mean_test_loss += nd.mean(test_loss).asscalar()

            print("Epoch %d, train average loss: %f" % (e, mean_train_loss / self.__num_train))
            print("Epoch %d, test  average loss: %f" % (e, mean_test_loss / self.__num_test))


if __name__ == "__main__":
    ls = L2Scratch(num_train=5, num_test=100, num_inputs=20, learning_rate=0.1, epochs=10, lamda=0)
    ls.data_prepare()
    ls.train_model()