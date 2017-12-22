# coding:utf-8

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd


class Scratch(object):

    def __init__(self, *, batch_size, learning_rate, epochs):
        # data prepare
        self.__num_output = 10
        self.__num_input = 784
        self.__w = None
        self.__b = None
        # 原始数据包含 X, y
        self.__train = None
        self.__test = None
        self.__w = None
        self.__b = None
        self.__params = None

        # function set

        # goodness of function loss function

        # goodness of function optimizer data
        self.__batch_size = batch_size
        self.__train_data_iter = None
        self.__test_data_iter = None

        # goodness of function optimizer function
        self.__learning_rate = learning_rate

        # pick the best function 模型训练
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
        z = nd.dot(self.__batch_X.reshape((-1, self.__num_input)), self.__w) + self.__b
        exp_z = nd.exp(z)
        partition = exp_z.sum(axis=1, keepdims=True)

        return exp_z / partition

    def goodness_of_function_loss_function(self):
        return - nd.log(nd.pick(self.__batch_y_hat, self.__batch_y))

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
    s = Scratch(batch_size=256, learning_rate=0.1, epochs=5)
    s.data_prepare()
    s.goodness_of_function_optimizer_data()
    s.train_model()