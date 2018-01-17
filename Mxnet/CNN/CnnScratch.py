# coding:utf-8

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class CnnScratch(object):

    def __init__(self, *, batch_size, learning_rate, epochs):
        # set_ctx
        self.__ctx = None

        # data prepare
        self.__train = None
        self.__test = None
        ## Conv
        self.__W1, self.__b1, self.__W2, self.__b2 = [None for _ in range(4)]
        ## Dense
        self.__W3, self.__b3, self.__W4, self.__b4 = [None for _ in range(4)]
        self.__params = None

        # function set

        # goodness of function loss function

        # goodness_of_function_optimizer_data
        self.__batch_size = batch_size
        self.__train_data_iter = None
        self.__test_data_iter = None

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

        weight_scale = .01

        # output channels = 20, kernel = (5,5)
        self.__W1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=self.__ctx)
        self.__b1 = nd.zeros(self.__W1.shape[0], ctx=self.__ctx)

        # output channels = 50, kernel = (3,3)
        self.__W2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=self.__ctx)
        self.__b2 = nd.zeros(self.__W2.shape[0], ctx=self.__ctx)

        # output dim = 128
        self.__W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=self.__ctx)
        self.__b3 = nd.zeros(self.__W3.shape[1], ctx=self.__ctx)

        # output dim = 10
        self.__W4 = nd.random_normal(shape=(self.__W3.shape[1], 10), scale=weight_scale, ctx=self.__ctx)
        self.__b4 = nd.zeros(self.__W4.shape[1], ctx=self.__ctx)

        self.__params = [self.__W1, self.__b1, self.__W2, self.__b2, self.__W3, self.__b3, self.__W4, self.__b4]

    def function_set(self):
        # 第一层卷积
        # 卷积
        h1_conv = nd.Convolution(
            data=self.__batch_X, weight=self.__W1, bias=self.__b1, kernel=self.__W1.shape[2:], num_filter=self.__W1.shape[0])
        # 激活
        h1_activation = nd.relu(h1_conv)
        # 池化
        h1 = nd.Pooling(data=h1_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
        # 第二层卷积
        h2_conv = nd.Convolution(
            data=h1, weight=self.__W2, bias=self.__b2, kernel=self.__W2.shape[2:], num_filter=self.__W2.shape[0])
        h2_activation = nd.relu(h2_conv)
        h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
        h2 = nd.flatten(h2)
        # 第一层全连接
        h3_linear = nd.dot(h2, self.__W3) + self.__b3
        h3 = nd.relu(h3_linear)
        # 第二层全连接
        h4_linear = nd.dot(h3, self.__W4) + self.__b4

        # print("1st conv block:", h1.shape)
        # print("2nd conv block:", h2.shape)
        # print("1st dense:", h3.shape)
        # print("2nd dense:", h4_linear.shape)
        # print("output:", h4_linear)

        return h4_linear

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(
            self.__train, self.__batch_size, shuffle=False)
        self.__test_data_iter = gluon.data.DataLoader(
            self.__test, self.__batch_size, shuffle=False)

    def goodness_of_function_loss_function(self):
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        return loss(self.__batch_y_hat, self.__batch_y)

    def goodness_of_function_optimizer_function(self):
        for param in self.__params:
            param[:] = param - self.__learning_rate / self.__batch_size * param.grad

    def train_model(self):
        for param in self.__params:
            param.attach_grad()

        for e in range(self.__epochs):
            total_mean_loss = 0
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                self.__batch_X = self.__batch_X.reshape((-1, 1, 28, 28)).as_in_context(self.__ctx)
                self.__batch_y = self.__batch_y.reshape((-1, 1)).as_in_context(self.__ctx)
                print(self.__batch_X)
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    loss = self.goodness_of_function_loss_function()
                loss.backward()
                self.goodness_of_function_optimizer_function()

                total_mean_loss += nd.mean(loss).asscalar()
            print("Epoch %d, average loss: %f" % (e, total_mean_loss / len(self.__train_data_iter)))


if __name__ == "__main__":
    cs = CnnScratch(batch_size=256, learning_rate=0.1, epochs=5)
    cs.set_ctx()
    cs.data_prepare()
    cs.goodness_of_function_optimizer_data()
    cs.train_model()