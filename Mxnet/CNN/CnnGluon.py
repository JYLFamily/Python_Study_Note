# coding:utf-8

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class CnnGluon(object):

    def __init__(self, *, batch_size, learning_rate, epochs):
        # data prepare
        # 原始数据包含 X, y
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

    def data_prepare(self):
        def transform(data, label):
            return data.astype("float32") / 255, label.astype("float32")
        self.__train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
        self.__test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

    def function_set(self):
        self.__net = gluon.nn.Sequential()
        with self.__net.name_scope():
            self.__net.add(
                gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=2, strides=2),
                gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=2, strides=2),
                gluon.nn.Flatten(),
                gluon.nn.Dense(128, activation="relu"),
                gluon.nn.Dense(10))
        self.__net.initialize()

    def goodness_of_function_loss_function(self):
        self.__softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(
            self.__train, self.__batch_size, shuffle=False)
        self.__test_data_iter = gluon.data.DataLoader(
            self.__test, self.__batch_size, shuffle=False)

    def goodness_of_function_optimizer_function(self):
        self.__trainer = gluon.Trainer(self.__net.collect_params(), "sgd", {"learning_rate": self.__learning_rate})

    def train_model(self):
        for e in range(self.__epochs):
            total_loss = 0.
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                # self.__batch_X (batch_size, height, width, channels)
                # self.__batch_X (batch_size, channels, height, width) 才能够使用
                self.__batch_X = self.__batch_X.reshape((-1, 1, 28, 28))
                self.__batch_y = self.__batch_y.reshape((-1, 1))
                with autograd.record():
                    self.__batch_y_hat = self.__net(self.__batch_X)
                    loss = self.__softmax_cross_entropy(self.__batch_y_hat, self.__batch_y)
                loss.backward()
                self.__trainer.step(self.__batch_size)

                total_loss += nd.mean(loss).asscalar()
            print("Epoch %d, average loss: %f" % (e, total_loss / len(self.__train_data_iter)))

    def test_model(self):
        for self.__batch_X, self.__batch_y in self.__test_data_iter:
            print(self.__net(self.__batch_X).argmax(axis=1))

if __name__ == "__main__":
    cg = CnnGluon(batch_size=256, learning_rate=0.1, epochs=5)
    cg.data_prepare()
    cg.function_set()
    cg.goodness_of_function_loss_function()
    cg.goodness_of_function_optimizer_data()
    cg.goodness_of_function_optimizer_function()
    cg.train_model()
    # cg.test_model()