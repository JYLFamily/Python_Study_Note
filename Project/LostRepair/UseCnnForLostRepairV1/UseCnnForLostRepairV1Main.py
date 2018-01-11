# coding:utf-8

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from Project.LostRepair.UseCnnForLostRepairV1.LostRepairBlock import LostRepairBlock


class UseCnnForLostRepairV1Main(object):

    def __init__(self, *,
            input_path=None, sep=None, header=None, test_size=None, random_state=None,
            batch_size=None, learning_rate=None, epochs=None):

        # set ctx
        self.__ctx = None

        # data prepare
        self.__input_path = input_path
        self.__sep = sep
        self.__header = header
        self.__X = None
        self.__y = None
        self.__test_size = test_size
        self.__random_state = random_state
        self.__train, self.__train_label, self.__test, self.__test_label = [None for _ in range(4)]
        self.__dataset_train = None
        self.__dataset_test = None

        # function set
        self.__net = None

        # goodness of function loss function
        self.__logistic_loss = None

        # goodness of function optimizer data
        self.__batch_size = batch_size
        self.__data_iter_train = None
        self.__data_iter_test = None

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
            _ = nd.zeros(shape=(1, 1), ctx=self.__ctx)
        except:
            self.__ctx = mx.cpu()

    def data_prepare(self):
        self.__X = pd.read_csv(self.__input_path, sep=self.__sep, header=self.__header, usecols=list(range(1, 4)))
        self.__y = pd.read_csv(self.__input_path, sep=self.__sep, header=self.__header, usecols=[0])

        self.__train, self.__test, self.__train_label, self.__test_label = train_test_split(
            self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state
        )

        encoder = LabelEncoder().fit(self.__train[3])
        self.__train[3] = encoder.transform(self.__train[3])
        self.__test[3] = encoder.transform(self.__test[3])
        self.__train = self.__train.values
        self.__test = self.__test.values
        self.__train_label = self.__train_label.values
        self.__test_label = self.__test_label.values

        scaler = OneHotEncoder(sparse=False).fit(self.__train[:, 2].reshape((-1, 1)))
        self.__train = np.hstack((
            StandardScaler().fit_transform(self.__train[:, [0, 1]]),
            scaler.transform(self.__train[:, 2].reshape((-1, 1))))
        )
        self.__test = np.hstack((
            StandardScaler().fit_transform(self.__test[:, [0, 1]]),
            scaler.transform(self.__test[:, 2].reshape((-1, 1))))
        )

        self.__dataset_train = gluon.data.ArrayDataset(nd.array(self.__train), nd.array(self.__train_label))
        self.__dataset_test = gluon.data.ArrayDataset(nd.array(self.__test), nd.array(self.__test_label))

    def function_set(self):
        self.__net = LostRepairBlock()
        self.__net.initialize(ctx=self.__ctx)

    def goodness_of_function_loss_function(self):
        self.__logistic_loss = gluon.loss.LogisticLoss()

    def goodness_of_function_optimizer_data(self):
        self.__data_iter_train = gluon.data.DataLoader(
            self.__dataset_train, self.__batch_size, shuffle=True
        )
        self.__data_iter_test = gluon.data.DataLoader(
            self.__dataset_test, self.__batch_size, shuffle=True
        )

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
            for self.__batch_X, self.__batch_y in self.__data_iter_train:
                self.__batch_X = self.__batch_X.as_in_context(self.__ctx)
                self.__batch_y = self.__batch_y.reshape((-1, 1)).as_in_context(self.__ctx)
                with autograd.record():
                    self.__batch_y_hat = self.__net(self.__batch_X)
                    loss = self.__logistic_loss(self.__batch_y_hat, self.__batch_y)
                loss.backward()
                self.__trainer.step(self.__batch_size)

                train_loss += nd.mean(loss).asscalar()
                train_acc += accuracy(self.__batch_y_hat, self.__batch_y)
            test_acc = evaluate_accuracy(self.__data_iter_test, self.__net, self.__ctx)
            print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
                e, train_loss / len(self.__data_iter_train), train_acc / len(self.__data_iter_train), test_acc))

if __name__ == "__main__":
    m = UseCnnForLostRepairV1Main(
        input_path="C:\\Users\\Dell\\Desktop\\model.txt",
        sep="\t",
        header=None,
        test_size=0.2,
        random_state=9,
        batch_size=64,
        learning_rate=0.1,
        epochs=20
    )
    m.set_ctx()
    m.data_prepare()
    m.function_set()
    m.goodness_of_function_loss_function()
    m.goodness_of_function_optimizer_data()
    m.goodness_of_function_optimizer_function()
    m.pick_the_best_function()