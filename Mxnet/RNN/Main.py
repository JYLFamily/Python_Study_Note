#  coding:utf-8

import math
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from Mxnet.RNN.LoadData import LoadData


class Main(object):
    def __init__(self, *, batch_size, num_steps, learning_rate, epochs, clipping_theta):
        # ------ set ctx ------
        self.__ctx = None

        # ------ data prepare ------
        # 语言数据
        self.__ld = None
        self.__vocab_size = None
        self.__corpus_indices = None
        # 隐藏层
        self.__W_xh, self.__W_hh, self.__b_h = [None for _ in range(3)]
        # 输出层
        self.__W_hy, self.__b_y = [None for _ in range(2)]
        # 网络参数
        self.__params = None

        # ------ function set ------

        # ------ goodness of function loss function ------

        # ------ goodness of function optimizer data ------
        self.__batch_size = batch_size
        self.__num_steps = num_steps

        # ------ goodness of function optimizer function ------
        self.__learning_rate = learning_rate

        # ------ pick the best function ------
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None
        self.__state = None
        self.__clipping_theta = clipping_theta

    def set_ctx(self):
        try:
            self.__ctx = mx.gpu()
            _ = nd.zeros((1, ), ctx=self.__ctx)
        except:
            self.__ctx = mx.cpu()

    def data_prepare(self):
        # 语言数据
        self.__ld = LoadData(zip_file_name="jaychou_lyrics.zip", txt_file_name="jaychou_lyrics.txt")
        self.__ld.set_data()
        self.__vocab_size = len(self.__ld.set_get_dict())
        self.__corpus_indices = self.__ld.set_get_index()

        # 神经网络参数
        input_dim = self.__vocab_size
        hidden_dim = 256
        output_dim = self.__vocab_size
        std = .01
        # 隐含层
        self.__W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=self.__ctx)
        self.__W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=self.__ctx)
        self.__b_h = nd.zeros(hidden_dim, ctx=self.__ctx)
        # 输出层
        self.__W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=self.__ctx)
        self.__b_y = nd.zeros(output_dim, ctx=self.__ctx)

        self.__params = [self.__W_xh, self.__W_hh, self.__b_h, self.__W_hy, self.__b_y]

    def function_set(self):
        self.__batch_y_hat = []
        for X in self.__batch_X:
            # X     batch_size × 字典长度
            # W_xh  字典长度 × hidden_num
            # state batch_size × hidden_num
            # W_hh  hidden_num × hidden_num
            # b_h   1 × hidden_num (广播)
            self.__state = nd.tanh(nd.dot(X, self.__W_xh) + nd.dot(self.__state, self.__W_hh) + self.__b_h)
            # W_hy  hidden_num × 字典长度
            # b_y   1 × 字典长度 (广播)
            self.__batch_y_hat.append(nd.dot(self.__state, self.__W_hy) + self.__b_y)
            # batch_y_hat (batch_size × num_steps, 字典长度)
            self.__batch_y_hat = nd.concat(*self.__batch_y_hat, dim=0)

    def goodness_of_loss_function(self):
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        return loss(self.__batch_y_hat, self.__batch_y)

    def goodness_of_function_optimizer_function(self):
        for param in self.__params:
            param[:] = param - self.__learning_rate / self.__batch_size * param.grad

    def goodness_of_function_optimizer_data(self):
        """相邻批量采样"""
        corpus_indices = nd.array(self.__corpus_indices, ctx=self.__ctx)
        data_len = len(self.__corpus_indices)
        batch_len = data_len // self.__batch_size

        indices = corpus_indices[0: self.__batch_size * batch_len].reshape((self.__batch_size, batch_len))
        epoch_size = (batch_len - 1) // self.__num_steps

        for i in range(epoch_size):
            i = i * self.__num_steps
            data = indices[:, i: i + self.__num_steps]
            label = indices[:, i + 1: i + self.__num_steps + 1]
            yield data, label

    def pick_the_best_function(self):
        def _grad_clipping(params, theta, ctx):
            if theta is not None:
                norm = nd.array([0.0], ctx)
                for p in params:
                    norm += nd.sum(p.grad ** 2)
                norm = nd.sqrt(norm).asscalar()
                if norm > theta:
                    for p in params:
                        p.grad[:] *= theta / norm

        for param in self.__params:
            param.attach_grad()

        for e in range(1, self.__epochs + 1):
            train_loss, num_examples = 0, 0
            self.__state = nd.zeros(shape=(self.__batch_size, 256), ctx=self.__ctx)

            for self.__batch_X, self.__batch_y in self.goodness_of_function_optimizer_data():
                self.__batch_X = [nd.one_hot(X, self.__vocab_size) for X in self.__batch_X.T]
                # batch_y (batch_size × num_steps, )
                # (第 0 时刻 batch_size 中第 1 样本 , 第 0 时刻 batch_size 中第 2 样本 , …… ,
                #  第 1 时刻 batch_size 中第 1 样本 , …… )
                self.__batch_y = self.__batch_y.T.reshape((-1, ))
                with autograd.record():
                    self.function_set()
                    loss = self.goodness_of_loss_function()
                loss.backward()

                _grad_clipping(self.__params, self.__clipping_theta, self.__ctx)
                self.goodness_of_function_optimizer_function()

                train_loss += nd.sum(loss).asscalar()
                num_examples += loss.size

            print("Epoch %d. Perplexity %f" % (e, math.exp(train_loss / num_examples)))


if __name__ == "__main__":
    m = Main(
        batch_size=32,
        num_steps=35,
        learning_rate=0.2,
        epochs=200,
        clipping_theta=5
    )
    m.set_ctx()
    m.data_prepare()
    m.pick_the_best_function()


