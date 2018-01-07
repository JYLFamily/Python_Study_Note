# coding:utf-8

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class DropoutScratch(object):

    def __init__(self, *, weight_scale, drop_prob1, drop_prob2, batch_size, learning_rate, epochs):
        # ctx
        self.__ctx = None

        # data prepare
        self.__weight_scale = weight_scale
        self.__train = None
        self.__test = None
        self.__num_inputs = 28 * 28
        self.__num_outputs = 10
        self.__num_hidden1 = 256
        self.__num_hidden2 = 256
        self.__W1, self.__b1, self.__W2, self.__b2, self.__W3, self.__b3, self.__params = [None for _ in range(7)]

        # function set
        self.__drop_prob1 = drop_prob1
        self.__drop_prob2 = drop_prob2

        # goodness of function loss function


        # goodness of function optimizer data
        self.__batch_size = batch_size
        self.__train_data_iter, self.__test_data_iter = [None for _ in range(2)]

        # goodness of function optimizer function
        self.__learning_rate = learning_rate

        # pick the best function
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def set_ctx(self):
        try:
            self.__ctx = mx.gpu()
            _ = nd.zeros((1,), ctx=self.__ctx)
        except:
            self.__ctx = mx.cpu()

    def data_prepare(self):
        def transform(data, label):
            return data.astype("float32") / 255, label.astype("float32")
        self.__train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
        self.__test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

        self.__W1 = nd.random_normal(
            shape=(self.__num_inputs, self.__num_hidden1), scale=self.__weight_scale, ctx=self.__ctx)
        self.__b1 = nd.zeros(shape=(1, self.__num_hidden1), ctx=self.__ctx)
        self.__W2 = nd.random_normal(
            shape=(self.__num_hidden1, self.__num_hidden2), scale=self.__weight_scale, ctx=self.__ctx)
        self.__b2 = nd.zeros(shape=(1, self.__num_hidden2), ctx=self.__ctx)
        self.__W3 = nd.random_normal(
            shape=(self.__num_hidden2, self.__num_outputs), scale=self.__weight_scale, ctx=self.__ctx)
        self.__b3 = nd.zeros(shape=(1, self.__num_outputs), ctx=self.__ctx)
        self.__params = [self.__W1, self.__b1, self.__W2, self.__b2, self.__W3, self.__b3]

    def function_set(self):
        def dropout(batch_X, drop_probability):
            keep_probability = 1 - drop_probability
            assert 0 <= keep_probability <= 1
            if keep_probability == 0:
                return batch_X.zeros_like()

            # > 保存的概率才能够保留该样本该神经元的输出
            mask = nd.random_uniform(
                0, 1.0, batch_X.shape, ctx=batch_X.context) < keep_probability
            # 保证 E[dropout(batch_X)] == batch_X
            scale = 1 / keep_probability

            return mask * batch_X * scale

        # Dense 需要 dropout Conv 其实不需要因为已经 share weight 了
        h1 = dropout(
            nd.relu(nd.dot(self.__batch_X.reshape((-1, self.__num_inputs)), self.__W1) + self.__b1), self.__drop_prob1)
        h2 = dropout(
            nd.relu(nd.dot(h1, self.__W2) + self.__b2), self.__drop_prob2)

        return nd.dot(h2, self.__W3) + self.__b3

    def goodness_of_function_loss_function(self):
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        return loss(self.__batch_y_hat, self.__batch_y)

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(
            self.__train, self.__batch_size, shuffle=False)
        self.__test_data_iter = gluon.data.DataLoader(
            self.__test, self.__batch_size, shuffle=False)

    def goodness_of_function_optimizer_function(self):
        for param in self.__params:
            param[:] = param - self.__learning_rate / self.__batch_size * param.grad

    def train_model(self):
        for param in self.__params:
            param.attach_grad()

        for e in range(self.__epochs):
            total_mean_loss = 0
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                self.__batch_X = self.__batch_X.reshape((-1, self.__num_inputs)).as_in_context(self.__ctx)
                self.__batch_y = self.__batch_y.reshape((-1, 1)).as_in_context(self.__ctx)
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    train_loss = self.goodness_of_function_loss_function()
                train_loss.backward()
                self.goodness_of_function_optimizer_function()
                total_mean_loss += nd.mean(train_loss).asscalar()

            print("Epoch %d, average train loss: %f" % (e, total_mean_loss / len(self.__train_data_iter)))


if __name__ == "__main__":
    ds = DropoutScratch(
        weight_scale=0.01, drop_prob1=0.5, drop_prob2=0.2, batch_size=256, learning_rate=0.1, epochs=5)
    ds.set_ctx()
    ds.data_prepare()
    ds.goodness_of_function_optimizer_data()
    ds.train_model()