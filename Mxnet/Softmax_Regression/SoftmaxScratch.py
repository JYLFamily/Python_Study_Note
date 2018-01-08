# coding:utf-8

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd


class SoftmaxScratch(object):

    def __init__(self, *, batch_size, learning_rate, epochs):
        # data prepare
        # 原始数据包含 X, y
        self.__num_output = 10
        self.__num_input = 784
        self.__train = None
        self.__test = None

        self.__w = None
        self.__b = None
        self.__params = None

        # function set

        # goodness of function loss function
        self.__batch_y_hat_exp = None
        self.__batch_y_hat_partition = None
        self.__batch_y_hat_exp_divided_partition = None

        # goodness of function optimizer data
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

    def data_prepare(self):
        def transform(data, label):
            return data.astype("float32") / 255, label.astype("float32")
        self.__train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
        self.__test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

        # 10 分类问题相当于有 10 个 Logistics Regression self.__num_output
        # 每个 Logistics Regression 接收 784 个特征 self.__num_input
        self.__w = nd.random_normal(shape=(self.__num_input, self.__num_output))
        self.__b = nd.random_normal(shape=(1, self.__num_output))
        self.__params = [self.__w, self.__b]

    def function_set(self):
        return nd.dot(self.__batch_X.reshape((-1, self.__num_input)), self.__w) + self.__b

    def goodness_of_function_loss_function(self):
        # 取指数使得所有值 > 0
        self.__batch_y_hat_exp = nd.exp(self.__batch_y_hat)
        # 求 partition 用于归一化概率
        self.__batch_y_hat_partition = self.__batch_y_hat_exp.sum(axis=1, keepdims=True)
        self.__batch_y_hat_exp_divided_partition = self.__batch_y_hat_exp / self.__batch_y_hat_partition

        return - nd.log(nd.pick(self.__batch_y_hat_exp_divided_partition, self.__batch_y))

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
            total_loss = 0
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    loss = self.goodness_of_function_loss_function()
                loss.backward()
                self.goodness_of_function_optimizer_function()
                total_loss += nd.mean(loss).asscalar()

            print("Epoch %d, average loss: %f" % (e, total_loss / len(self.__train_data_iter)))


if __name__ == "__main__":
    ss = SoftmaxScratch(batch_size=256, learning_rate=0.1, epochs=5)
    ss.data_prepare()
    ss.goodness_of_function_optimizer_data()
    ss.train_model()