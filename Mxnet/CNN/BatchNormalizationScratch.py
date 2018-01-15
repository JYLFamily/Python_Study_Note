# coding:utf-8

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class BatchNormalizationScratch(object):
    def __init__(self, *, batch_size=None, learning_rate=None, epochs=None):
        # set_ctx
        self.__ctx = None

        # data prepare
        self.__train = None
        self.__test = None
        ## Conv
        self.__W1, self.__b1, self.__W2, self.__b2 = [None for _ in range(4)]
        ## Dense
        self.__W3, self.__b3, self.__W4, self.__b4 = [None for _ in range(4)]
        ## BN
        self.__gamma1, self.__beta1, self.__moving_mean1, self.__moving_variance1 = [None for _ in range(4)]
        self.__gamma2, self.__beta2, self.__moving_mean2, self.__moving_variance2 = [None for _ in range(4)]

        self.__params = None

        # function set

        # goodness of function loss function

        # goodness_of_function_optimizer_data
        self.__batch_size = batch_size
        self.__train_data_iter = None
        self.__test_data_iter = None

        # goodness of function optimizer function
        self.__learning_rate = learning_rate

        # pick the best function
        self.__is_training = None
        self.__epochs = epochs
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def set_ctx(self):
        try:
            self.__ctx = mx.gpu()
            _ = nd.zeros((1,), ctx=self.__ctx)
        except:
            self.__ctx = mx.cpu()

    def data_prepare(self):
        def transform(data, label):
            return data.astype("float32") / 255, label.astype("float32")
        self.__train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
        self.__test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

        weight_scale = .01

        # 第一层卷积 output_channels/num_filter = 20 kernel size (5, 5)
        self.__W1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=self.__ctx)
        self.__b1 = nd.zeros(20, ctx=self.__ctx)

        # 第一层批量归一化 ,
        # 卷积批量归一化是针对每一个 channels 所以卷积之后一个样本有 20 个 channels 每个 channels 要 × gamma + beta
        # 全连接批量归一化是针对每一个 feature 因为全连接 feature 个数可以类比为卷积后 channels 个数 , 所以 shape 也是如此
        self.__gamma1 = nd.random_normal(shape=20, scale=weight_scale, ctx=self.__ctx)
        self.__beta1 = nd.random_normal(shape=20, scale=weight_scale, ctx=self.__ctx)
        self.__moving_mean1 = nd.zeros(20, ctx=self.__ctx)
        self.__moving_variance1 = nd.zeros(20, ctx=self.__ctx)

        # 第二层卷积 output_channels/num_filter = 50 kernel size (3, 3)
        self.__W2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=self.__ctx)
        self.__b2 = nd.zeros(50, ctx=self.__ctx)

        # 第二层批量归一化
        self.__gamma2 = nd.random_normal(shape=50, scale=weight_scale, ctx=self.__ctx)
        self.__beta2 = nd.random_normal(shape=50, scale=weight_scale, ctx=self.__ctx)
        ## 这里 shape 为什么是 50 ? 相当于每个 channels 有一个均值。
        self.__moving_mean2 = nd.zeros(50, ctx=self.__ctx)
        self.__moving_variance2 = nd.zeros(50, ctx=self.__ctx)

        # 第一个 Dense Flatten 之后是 1250 个 feature
        self.__W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=self.__ctx)
        self.__b3 = nd.random_normal(shape=(1, 128), ctx=self.__ctx)

        # 第二个 Dense 输出层 输出到 Softmax
        self.__W4 = nd.random_normal(shape=(128, 10), scale=weight_scale, ctx=self.__ctx)
        self.__b4 = nd.zeros(shape=(1, 10), ctx=self.__ctx)

        self.__params = [self.__W1, self.__b1, self.__gamma1, self.__beta1, self.__W2, self.__b2,
                         self.__gamma2, self.__beta2, self.__W3, self.__b3, self.__W4, self.__b4]

    def function_set(self):
        def batch_norm(X, gamma, beta, is_training, moving_mean, moving_variance, eps=1e-5, moving_momentum=0.9):
            assert len(X.shape) in (2, 4)
            # 全连接: batch_size x feature
            if len(X.shape) == 2:
                # 每个输入维度在样本上的平均和方差
                mean = X.mean(axis=0)
                variance = ((X - mean) ** 2).mean(axis=0)
            # 2D卷积: batch_size x channel x height x width
            else:
                # 对每个通道算均值和方差，需要保持 4D 形状使得可以正确的广播
                mean = X.mean(axis=(0, 2, 3), keepdims=True)
                variance = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
                # 变形使得可以正确的广播
                moving_mean = moving_mean.reshape(mean.shape)
                moving_variance = moving_variance.reshape(mean.shape)

            # 均一化
            if is_training:
                X_hat = (X - mean) / nd.sqrt(variance + eps)
                # !!! 更新全局的均值和方差
                # 每一个 batch_X 都会使用上个 batch_X 的 0.9 与 这个 batch_X 的 0.1
                moving_mean[:] = moving_momentum * moving_mean + (1.0 - moving_momentum) * mean
                moving_variance[:] = moving_momentum * moving_variance + (1.0 - moving_momentum) * variance
            else:
                # !!! 测试阶段使用全局的均值和方差
                X_hat = (X - moving_mean) / nd.sqrt(moving_variance + eps)

            # 拉升和偏移
            return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)

        # 第一层卷积
        h1_conv = nd.Convolution(
            data=self.__batch_X, weight=self.__W1, bias=self.__b1, kernel=(5, 5), num_filter=20)
        # 第一个 BN
        h1_bn = batch_norm(
            h1_conv, self.__gamma1, self.__beta1, self.__is_training, self.__moving_mean1, self.__moving_variance1)
        h1_activation = nd.relu(h1_bn)
        h1 = nd.Pooling(
            data=h1_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))

        # 第二层卷积
        h2_conv = nd.Convolution(
            data=h1, weight=self.__W2, bias=self.__b2, kernel=(3, 3), num_filter=50)
        # 第二个 BN
        h2_bn = batch_norm(
            h2_conv, self.__gamma2, self.__beta2, self.__is_training, self.__moving_mean2, self.__moving_variance2)
        h2_activation = nd.relu(h2_bn)
        h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
        h2 = nd.flatten(h2)

        # 第一层全连接
        h3_linear = nd.dot(h2, self.__W3) + self.__b3
        h3 = nd.relu(h3_linear)

        # 第二层全连接
        h4_linear = nd.dot(h3, self.__W4) + self.__b4

        return h4_linear

    def goodness_of_function_optimizer_data(self):
        self.__train_data_iter = gluon.data.DataLoader(
            self.__train, self.__batch_size, shuffle=False)
        self.__test_data_iter = gluon.data.DataLoader(
            self.__test, self.__batch_size, shuffle=False)

    def goodness_of_function_loss_function(self):
        loss = gluon.loss.SoftmaxCrossEntropyLoss()

        return loss(self.__batch_y_hat, self.__batch_y)

    def goodness_of_function_optimizer_function(self):
        for param in self.__params:
            param[:] = param - self.__learning_rate / self.__batch_size * param.grad

    def pick_the_best_function(self):
        for param in self.__params:
            param.attach_grad()

        def accuracy(y_hat, y):
            # 注意这里 y_hat 的 shape 必须与 y 的 shape 保持一致
            y_hat = y_hat.as_in_context(self.__ctx)
            y = y.as_in_context(self.__ctx)
            return nd.mean(y_hat.argmax(axis=1).reshape(y.shape) == y).asscalar()

        def evaluate_accuracy(data_iter, net, ctx):
            acc = 0.
            for batch_X, batch_y in data_iter:
                self.__batch_X = batch_X.reshape((-1, 1, 28, 28)).as_in_context(ctx)
                self.__batch_y = batch_y.reshape((-1, 1)).as_in_context(ctx)
                batch_y_hat = net()
                acc += accuracy(batch_y_hat, batch_y)
            return acc / len(data_iter)

        for e in range(self.__epochs):
            train_loss = 0.
            train_acc = 0.
            self.__is_training = True
            for self.__batch_X, self.__batch_y in self.__train_data_iter:
                self.__batch_X = self.__batch_X.reshape((-1, 1, 28, 28)).as_in_context(self.__ctx)
                self.__batch_y = self.__batch_y.reshape((-1, 1)).as_in_context(self.__ctx)
                with autograd.record():
                    self.__batch_y_hat = self.function_set()
                    loss = self.goodness_of_function_loss_function()
                loss.backward()
                self.goodness_of_function_optimizer_function()

                train_loss += nd.mean(loss).asscalar()
                train_acc += accuracy(self.__batch_y_hat, self.__batch_y)
            self.__is_training = False
            test_acc = evaluate_accuracy(self.__test_data_iter, self.function_set, self.__ctx)
            print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
                e, train_loss / len(self.__train_data_iter), train_acc / len(self.__train_data_iter), test_acc))


if __name__ == "__main__":
    bns = BatchNormalizationScratch(batch_size=256, learning_rate=0.1, epochs=5)
    bns.set_ctx()
    bns.data_prepare()
    bns.goodness_of_function_optimizer_data()
    bns.pick_the_best_function()