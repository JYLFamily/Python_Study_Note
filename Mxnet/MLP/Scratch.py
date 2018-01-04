# coding:utf-8

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class Scratch(object):

    def __init__(self, *, num_hidden, batch_size, learning_rate, epochs):
        # data prepare
        self.__num_output = 10
        self.__num_input = 784
        self.__train = None
        self.__test = None

        # parameters
        self.__num_hidden = num_hidden
        self.__w1 = None
        self.__b1 = None
        self.__w2 = None
        self.__b2 = None
        self.__params = None

        # function set & goodness of function loss function

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

        # loc scale 分布的均值与方差 , 方差越小越容易收敛
        self.__w1 = nd.random_normal(shape=(self.__num_input, self.__num_hidden), scale=0.01)
        self.__b1 = nd.random_normal(shape=(1, self.__num_hidden))
        self.__w2 = nd.random_normal(shape=(self.__num_hidden, self.__num_output), scale=0.01)
        self.__b2 = nd.random_normal(shape=(1, self.__num_output))
        self.__params = [self.__w1, self.__b1, self.__w2, self.__b2]

    def function_set(self):
        # relu = lambda x: nd.maximum(x, 0)

        def relu(x):
            return nd.maximum(x, 0)

        hidden_layer_before_act = nd.dot(self.__batch_X.reshape((-1, self.__num_input)), self.__w1) + self.__b1
        hidden_layer_after_act = relu(hidden_layer_before_act)
        output_layer_before_act = nd.dot(hidden_layer_after_act, self.__w2) + self.__b2

        return output_layer_before_act

    def goodness_of_function_loss_function(self):
        loss = gluon.loss.SoftmaxCELoss()

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
    s = Scratch(num_hidden=256, batch_size=256, learning_rate=0.5, epochs=5)
    s.data_prepare()
    s.goodness_of_function_optimizer_data()
    s.train_model()