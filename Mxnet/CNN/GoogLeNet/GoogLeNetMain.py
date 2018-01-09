# coding:utf-8

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import image
from mxnet import init
from mxnet import gluon
from Mxnet.CNN.GoogLeNet.GoogLeNet import GoogLeNet


class GoogLeNetMain(object):

    def __init__(self, *, batch_size, learning_rate, epochs):
        # set ctx
        self.__ctx = None

        # data prepare
        self.__resize = 96
        self.__train = None
        self.__test = None

        # function set
        self.__net = None

        # goodness of function loss function
        self.__softmax_cross_entropy = None

        # goodness of function optimizer data
        self.__batch_size = batch_size
        self.__train_data_iter = None
        self.__test_data_iter = None

        # goodness of function optimizer function
        self.__learning_rate = learning_rate
        self.__trainer = None

        # pick the best function
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def set_ctx(self):
        try:
            self.__ctx = mx.gpu()
            _ = nd.zeros(shape=(1,), ctx=self.__ctx)
        except:
            self.__ctx = mx.cpu()

    def data_prepare(self):
        def transform_mnist(data, label):
            # transform a batch of examples
            if self.__resize:
                # data 默认 (28, 28, 1)
                # data imresize 后 (224, 224, 1)
                data = image.imresize(data, self.__resize, self.__resize)
            # change data from height x weight x channel to channel x height x weight
            return nd.transpose(data.astype("float32"), (2, 0, 1)) / 255, label.astype("float32")

        self.__train = gluon.data.vision.FashionMNIST(train=True, transform=transform_mnist)
        self.__test = gluon.data.vision.FashionMNIST(train=False, transform=transform_mnist)

    def function_set(self):
        self.__net = GoogLeNet(num_classes=10, verbose=False)
        self.__net.initialize(init=init.Xavier(), ctx=self.__ctx)

    def goodness_of_function_loss_function(self):
        self.__softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(
            self.__train, self.__batch_size, shuffle=True)
        self.__test_data_iter = gluon.data.DataLoader(
            self.__test, self.__batch_size, shuffle=False)

    def goodness_of_function_optimizer_function(self):
        self.__trainer = gluon.Trainer(
            self.__net.collect_params(), "sgd", {"learning_rate": self.__learning_rate})

    def pick_the_best_function(self):
        def accuracy(y_hat, y):
            # 注意这里 y_hat 的 shape 必须与 y 的 shape 保持一致
            return nd.mean(y_hat.argmax(axis=1).reshape(y.shape) == y).asscalar()

        def evaluate_accuracy(data_iter, net, ctx):
            acc = 0.
            for batch_X, batch_y in data_iter:
                batch_X = batch_X.as_in_context(ctx)
                batch_y = batch_y.as_in_context(ctx)
                batch_y = batch_y.reshape((-1, 1))
                batch_y_hat = net(batch_X)
                acc += accuracy(batch_y_hat, batch_y)
            return acc / len(data_iter)

        for e in range(self.__epochs):
            train_loss = 0.
            train_acc = 0.
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                self.__batch_X = self.__batch_X.as_in_context(self.__ctx)
                self.__batch_y = self.__batch_y.reshape((-1, 1)).as_in_context(self.__ctx)
                with autograd.record():
                    self.__batch_y_hat = self.__net(self.__batch_X)
                    loss = self.__softmax_cross_entropy(self.__batch_y_hat, self.__batch_y)
                loss.backward()
                self.__trainer.step(self.__batch_size)

                train_loss += nd.mean(loss).asscalar()
                train_acc += accuracy(self.__batch_y_hat, self.__batch_y)
            test_acc = evaluate_accuracy(self.__test_data_iter, self.__net, self.__ctx)
            print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
                e, train_loss / len(self.__train_data_iter), train_acc / len(self.__train_data_iter), test_acc))


if __name__ == "__main__":
    gnm = GoogLeNetMain(batch_size=64, learning_rate=0.1, epochs=10)
    gnm.set_ctx()
    gnm.data_prepare()
    gnm.function_set()
    gnm.goodness_of_function_loss_function()
    gnm.goodness_of_function_optimizer_data()
    gnm.goodness_of_function_optimizer_function()
    gnm.pick_the_best_function()