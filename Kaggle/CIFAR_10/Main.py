# coding:utf-8

import datetime
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from Kaggle.CIFAR_10.OptimizerData import OptimizerData
from Kaggle.CIFAR_10.ResNet import ResNet


class Main(object):

    def __init__(self, *,
                 input_path, folder_list, batch_size, learning_rate, momentum, wd, leanring_rate_decay, epochs):
        # set ctx
        self.__ctx = None

        # data prepare

        # function set
        self.__net = None

        # goodness of function loss function
        self.__loss = None

        # goodness of function optimizer data
        self.__input_path = input_path
        self.__folder_list = folder_list
        self.__batch_size = batch_size

        self.__train_data_iter = None
        self.__valid_data_iter = None
        self.__train_valid_data_iter = None
        self.__test_data_iter = None

        # goodness of function optimizer function
        self.__trainer = None
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__wd = wd

        # pick the best function
        self.__learning_rate_decay = leanring_rate_decay
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def set_ctx(self):
        try:
            self.__ctx = mx.gpu()
            _ = nd.zeros(shape=(1, ), ctx=self.__ctx)
        except:
            self.__ctx = mx.cpu()

    def data_prepare(self):
        pass

    def function_set(self):
        self.__net = ResNet(num_classes=10, verbose=False)
        self.__net.initialize(ctx=self.__ctx)
        # self.__net.hybridize()

    def goodness_of_function_loss_function(self):
        self.__loss = gluon.loss.SoftmaxCrossEntropyLoss()

    def goodness_of_function_optimizer_data(self):
        od = OptimizerData(input_path=self.__input_path, folder_list=self.__folder_list, batch_size=self.__batch_size)
        self.__train_data_iter, self.__valid_data_iter, self.__train_valid_data_iter, self.__test_data_iter = \
            od.load_data()

    def goodness_of_function_optimizer_function(self):
        self.__trainer = gluon.Trainer(
            self.__net.collect_params(),
            "sgd",
            {"learning_rate": self.__learning_rate,
             "momentum": self.__momentum,
             "wd": self.__wd}
        )

    def pick_the_best_function(self):
        def accuracy(y_hat, y):
            # 注意这里 y_hat 的 shape 必须与 y 的 shape 保持一致
            return nd.mean(y_hat.argmax(axis=1).reshape(y.shape) == y).asscalar()

        def evaluate_accuracy(data_iter, net, ctx):
            acc = 0.
            for batch_X, batch_y in data_iter:
                batch_X = batch_X.as_in_context(ctx)
                batch_y = batch_y.as_in_context(ctx)
                batch_y_hat = net(batch_X)
                acc += accuracy(batch_y_hat, batch_y)
            return acc / len(data_iter)

        prev_time = datetime.datetime.now()
        for epoch in range(self.__epochs):
            train_loss = 0.0
            train_acc = 0.0
            if epoch > 0 and epoch % self.__learning_rate_decay == 0:
                self.__trainer.set_learning_rate(self.__trainer.learning_rate * self.__learning_rate_decay)

            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                self.__batch_X = self.__batch_X.as_in_context(self.__ctx)
                self.__batch_y = self.__batch_y.as_in_context(self.__ctx)
                with autograd.record():
                    self.__batch_y_hat = self.__net(self.__batch_X)
                    loss = self.__loss(self.__batch_y_hat, self.__batch_y)
                loss.backward()
                self.__trainer.step(self.__batch_size)

                train_loss += nd.mean(loss).asscalar()
                train_acc += accuracy(self.__batch_y_hat, self.__batch_y)
            cur_time = datetime.datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)

            if self.__valid_data_iter is not None:
                valid_acc = evaluate_accuracy(self.__valid_data_iter, self.__net, self.__ctx)
                epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                             % (epoch, train_loss / len(self.__train_data_iter),
                                train_acc / len(self.__train_data_iter), valid_acc))
            else:
                epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                             % (epoch, train_loss / len(self.__train_data_iter),
                                train_acc / len(self.__train_data_iter)))
            prev_time = cur_time
            print(epoch_str + time_str + ", lr " + str(self.__trainer.learning_rate))


if __name__ == "__main__":
    m = Main(
        input_path="D:\\Code\\kaggle\\cifar10\\train_valid_test",
        folder_list=["train", "valid", "train_valid", "test"],
        batch_size=128,
        learning_rate=0.1,
        momentum=0.9,
        wd=5e-4,
        leanring_rate_decay=0.1,
        epochs=100
    )
    m.set_ctx()
    m.function_set()
    m.goodness_of_function_loss_function()
    m.goodness_of_function_optimizer_data()
    m.goodness_of_function_optimizer_function()
    m.pick_the_best_function()
