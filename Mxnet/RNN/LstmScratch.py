# coding:utf-8

#  coding:utf-8

import math
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from Mxnet.RNN.LoadData import LoadData


class GruScratch(object):
    def __init__(self, *, hidden_dim, batch_size, num_steps, learning_rate, epochs, clipping_theta, seqs):
        # ------ set ctx -----------
        self.__ctx = None

        # ------ data prepare ------
        # 语言数据
        self.__ld = None
        self.__char_to_idx = None
        self.__idx_to_char = None
        # 字典长度
        self.__vocab_size = None
        self.__corpus_indices = None

        # 隐藏层数目
        self.__hidden_dim = hidden_dim
        # 隐藏层
        # 输入门
        self.__W_xi, self.__W_hi, self.__b_i = [None for _ in range(3)]
        # 遗忘门
        self.__W_xf, self.__W_hf, self.__b_f = [None for _ in range(3)]
        # 输出门
        self.__W_xo, self.__W_ho, self.__b_o = [None for _ in range(3)]
        # 候选细胞
        self.__W_xc, self.__W_hc, self.__b_c = [None for _ in range(3)]
        # 输出层
        self.__W_hy, self.__b_y = [None for _ in range(2)]

        # 网络参数
        self.__params = None

        # ------ function set ------------------------------

        # ------ goodness of function loss function --------

        # ------ goodness of function optimizer data -------
        self.__batch_size = batch_size
        self.__num_steps = num_steps

        # ------ goodness of function optimizer function ---
        self.__learning_rate = learning_rate

        # ------ pick the best function --------------------
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None
        self.__state = None
        self.__C = None
        self.__clipping_theta = clipping_theta

        # ------ best function predict ---------------------
        self.__seqs = seqs

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
        self.__char_to_idx, self.__idx_to_char = self.__ld.set_get_dict()
        self.__vocab_size = len(self.__char_to_idx)
        self.__corpus_indices = self.__ld.set_get_index()

        # 神经网络参数
        input_dim = self.__vocab_size
        hidden_dim = self.__hidden_dim
        output_dim = self.__vocab_size
        std = .001
        # 隐含层
        # 输入门
        self.__W_xi = nd.random_normal(std, shape=(input_dim, hidden_dim), ctx=self.__ctx)
        self.__W_hi = nd.random_normal(std, shape=(hidden_dim, hidden_dim), ctx=self.__ctx)
        self.__b_i = nd.zeros(hidden_dim, ctx=self.__ctx)
        # 遗忘门
        self.__W_xf = nd.random_normal(std, shape=(input_dim, hidden_dim), ctx=self.__ctx)
        self.__W_hf = nd.random_normal(std, shape=(hidden_dim, hidden_dim), ctx=self.__ctx)
        self.__b_f = nd.zeros(hidden_dim, ctx=self.__ctx)
        # 输出门
        self.__W_xo = nd.random_normal(std, shape=(input_dim, hidden_dim), ctx=self.__ctx)
        self.__W_ho = nd.random_normal(std, shape=(hidden_dim, hidden_dim), ctx=self.__ctx)
        self.__b_o = nd.zeros(hidden_dim, ctx=self.__ctx)
        # 候选记忆细胞
        self.__W_xc = nd.random_normal(std, shape=(input_dim, hidden_dim), ctx=self.__ctx)
        self.__W_hc = nd.random_normal(std, shape=(hidden_dim, hidden_dim), ctx=self.__ctx)
        self.__b_c = nd.zeros(hidden_dim, ctx=self.__ctx)
        # 输出层
        self.__W_hy = nd.random_normal(std, shape=(hidden_dim, output_dim), ctx=self.__ctx)
        self.__b_y = nd.zeros(output_dim, ctx=self.__ctx)

        self.__params = [self.__W_xi, self.__W_hi, self.__b_i,
                         self.__W_xi, self.__W_hi, self.__b_i,
                         self.__W_xi, self.__W_hi, self.__b_i,
                         self.__W_hy, self.__b_y]

    def function_set(self):
        self.__batch_y_hat = []
        for X in self.__batch_X:
            I = nd.sigmoid(nd.dot(X, self.__W_xi) + nd.dot(self.__state, self.__W_hi) + self.__b_i)
            F = nd.sigmoid(nd.dot(X, self.__W_xf) + nd.dot(self.__state, self.__W_hf) + self.__b_f)
            O = nd.sigmoid(nd.dot(X, self.__W_xo) + nd.dot(self.__state, self.__W_ho) + self.__b_o)
            C_tilda = nd.tanh(nd.dot(X, self.__W_xc) + nd.dot(self.__state, self.__W_hc) + self.__b_c)
            # 注意这里 C 同 state 一样
            self.__C = F * self.__C + I * C_tilda
            self.__state = O * nd.tanh(self.__C)
            self.__batch_y_hat.append(nd.dot(self.__state, self.__W_hy) + self.__b_y)
        self.__batch_y_hat = nd.concat(*self.__batch_y_hat, dim=0)

        return self.__batch_y_hat

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
        for param in self.__params:
            param.attach_grad()

        def __grad_clipping(params, theta, ctx):
            if theta is not None:
                norm = nd.array([0.0], ctx)
                for p in params:
                    norm += nd.sum(p.grad ** 2)
                norm = nd.sqrt(norm).asscalar()
                if norm > theta:
                    for p in params:
                        p.grad[:] *= theta / norm

        for e in range(1, self.__epochs + 1):
            train_loss, num_examples = 0, 0
            self.__state = nd.zeros(shape=(self.__batch_size, self.__hidden_dim), ctx=self.__ctx)
            self.__C = nd.zeros(shape=(self.__batch_size, self.__hidden_dim), ctx=self.__ctx)

            for self.__batch_X, self.__batch_y in self.goodness_of_function_optimizer_data():
                # batch_X list len(list)==num_steps list 中每一个元素是一个时刻的 batch_size × 字典长度
                self.__batch_X = [nd.one_hot(X, self.__vocab_size) for X in self.__batch_X.T]
                # batch_y (batch_size × num_steps, )
                # (第 0 时刻 batch_size 中第 1 样本 , 第 0 时刻 batch_size 中第 2 样本 , …… ,
                # 第 1 时刻 batch_size 中第 1 样本 , …… )
                self.__batch_y = self.__batch_y.T.reshape((-1, ))
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    loss = self.goodness_of_loss_function()
                loss.backward()

                __grad_clipping(self.__params, self.__clipping_theta, self.__ctx)
                self.goodness_of_function_optimizer_function()

                train_loss += nd.sum(loss).asscalar()
                num_examples += loss.size

            print("Epoch %d. Perplexity %f" % (e, math.exp(train_loss / num_examples)))

    def best_function_predict(self):
        for prefix in self.__seqs:
            prefix = prefix.lower()
            self.__state = nd.zeros(shape=(1, self.__hidden_dim), ctx=self.__ctx)
            self.__C = nd.zeros(shape=(1, self.__hidden_dim), ctx=self.__ctx)
            output = [self.__char_to_idx[prefix[0]]]

            for i in range(100 + len(prefix)):
                self.__batch_X = nd.array([output[-1]], ctx=self.__ctx)
                self.__batch_X = nd.one_hot(self.__batch_X, self.__vocab_size)
                self.__batch_y_hat = self.function_set()
                if i < len(prefix) - 1:
                    next_input = self.__char_to_idx[prefix[i + 1]]
                else:
                    # 这里跟文档中不一致
                    next_input = int(self.__batch_y_hat[0].asnumpy().argmax())
                output.append(next_input)
            print("------------------------------------------------")
            print("".join([self.__idx_to_char[i] for i in output]))


if __name__ == "__main__":
    gs = GruScratch(
        hidden_dim=256,
        batch_size=32,
        num_steps=35,
        learning_rate=0.02,
        epochs=300,
        clipping_theta=1,
        seqs=["分开"]
    )
    gs.set_ctx()
    gs.data_prepare()
    gs.pick_the_best_function()
    gs.best_function_predict()