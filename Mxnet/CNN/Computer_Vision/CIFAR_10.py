# coding:utf-8

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import image
from mxnet import init
from mxnet import gluon


class CIFAR_10(object):
    def __init__(self, *, train_augs=None, test_augs=None, batch_size=None, learning_rate=None, epochs=None):
        # set ctx
        self.__ctx = None

        # data prepare
        self.__train_augs = train_augs
        self.__test_augs = test_augs
        self.__cifar10_train = None
        self.__cifar10_test = None

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

    def set_ctx(self):
        try:
            self.__ctx = mx.gpu()
            _ = nd.zeros(shape=(1, ), ctx=mx.gpu())
        except:
            self.__ctx = mx.cpu()

    # 如下的 data_prepare 方式更像是预处理不是图片增广 , 只是将原始样本水平旋转一下 , 然后随机剪裁变小
    # 一个样本还是一个样本我理解真正的图片增广应该是多种 处理后 concat channels 变多
    def data_prepare(self):
        def apply_aug_list(img, augs):
            for f in augs:
                img = f(img)
            return img

        def get_transform(augs):
            def transform(data, label):
                # data: height x width x channel
                data = data.astype("float32")
                if augs is not None:
                    data = apply_aug_list(data, augs)
                data = nd.transpose(data, (2, 0, 1)).clip(0, 255) / 255
                return data, label.astype("float32")
            return transform

        self.__cifar10_train = gluon.data.vision.CIFAR10(
            train=True, transform=get_transform(self.__train_augs))
        self.__cifar10_test = gluon.data.vision.CIFAR10(
            train=False, transform=get_transform(self.__test_augs))

    def function_set(self):
        self.__net = gluon.nn.Sequential()
        with self.__net.name_scope():
            self.__net.add(
                # 第一阶段
                gluon.nn.Conv2D(channels=96, kernel_size=3,
                          strides=1, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=3, strides=2),
                # 第二阶段
                gluon.nn.Conv2D(channels=256, kernel_size=3,
                          padding=1, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=3, strides=2),
                # 第三阶段
                gluon.nn.Conv2D(channels=384, kernel_size=3,
                          padding=1, activation='relu'),
                gluon.nn.Conv2D(channels=384, kernel_size=3,
                          padding=1, activation='relu'),
                gluon.nn.Conv2D(channels=256, kernel_size=3,
                          padding=1, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=3, strides=2),
                # 第四阶段
                gluon.nn.Flatten(),
                gluon.nn.Dense(4096, activation="relu"),
                gluon.nn.Dropout(.5),
                # 第五阶段
                gluon.nn.Dense(4096, activation="relu"),
                gluon.nn.Dropout(.5),
                # 第六阶段
                gluon.nn.Dense(10)
            )
            self.__net.initialize(init=init.Xavier(), ctx=self.__ctx)

    def goodness_of_function_loss_function(self):
        self.__softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(
            self.__cifar10_train, self.__batch_size, shuffle=True)
        self.__test_data_iter = gluon.data.DataLoader(
            self.__cifar10_test, self.__batch_size, shuffle=False)

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
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                self.__batch_X = self.__batch_X.as_in_context(self.__ctx)
                self.__batch_y = self.__batch_y.reshape((-1, 1)).as_in_context(self.__ctx)
                with autograd.record():
                    self.__batch_y_hat = self.__net(self.__batch_X)
                    loss = self.__softmax_cross_entropy(self.__batch_y_hat, self.__batch_y)
                loss.backward()
                self.__trainer.step(self.__batch_size)

                train_loss += nd.mean(loss).asscalar()
                train_acc += accuracy(self.__batch_y_hat, self.__batch_y)
            test_acc = evaluate_accuracy(self.__test_data_iter, self.__net, self.__ctx)
            print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
                e, train_loss / len(self.__train_data_iter), train_acc / len(self.__train_data_iter), test_acc))


if __name__ == "__main__":
    c10 = CIFAR_10(
        train_augs=[image.HorizontalFlipAug(.5), image.RandomCropAug((28, 28))],
        test_augs=[image.CenterCropAug((28, 28))],
        batch_size=128,
        learning_rate=0.1,
        epochs=5)
    c10.set_ctx()
    c10.data_prepare()
    c10.function_set()
    c10.goodness_of_function_loss_function()
    c10.goodness_of_function_optimizer_data()
    c10.goodness_of_function_optimizer_function()
    c10.pick_the_best_function()
