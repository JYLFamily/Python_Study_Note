# coding:utf-8

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class Gluon(object):

    def __init__(self):
        self.__num_inputs = 2
        self.__num_examples = 1000

        self.__true_w = nd.array([2, -3.4]).reshape((2, 1))
        self.__true_b = nd.array([4.2]).reshape((1, 1))

        self.__X = nd.random_normal(shape=(self.__num_examples, self.__num_inputs))
        self.__y = nd.dot(self.__X, self.__true_w) + self.__true_b
        self.__y += 0.01 * nd.random_normal(shape=self.__y.shape)

        self.__batch_size = 10
        self.__data_iter = None
        self.__net = None
        self.__loss_function = None
        self.__trainer = None

    def set_data_iter(self):
        self.__data_iter = (gluon.data.DataLoader(
            gluon.data.ArrayDataset(self.__X, self.__y), self.__batch_size, shuffle=True))

    def set_model(self):
        self.__net = gluon.nn.Sequential()
        self.__net.add(gluon.nn.Dense(1))
        self.__net.initialize()

    def set_loss_function(self):
        self.__loss_function = gluon.loss.L2Loss()

    def train_model(self):
        self.__trainer = gluon.Trainer(
            self.__net.collect_params(), "sgd", {"learning_rate": 0.1})

        epochs = 5
        batch_size = 10
        for e in range(epochs):
            total_loss = 0
            for data, label in self.__data_iter:
                with autograd.record():
                    output = self.__net(data)
                    loss = self.__loss_function(output, label)
                loss.backward()
                self.__trainer.step(batch_size)
                total_loss += nd.sum(loss).asscalar()
            print("Epoch %d, average loss: %f" % (e, total_loss / self.__num_examples))


if __name__ == "__main__":
    g = Gluon()
    g.set_data_iter()
    g.set_model()
    g.set_loss_function()
    g.train_model()