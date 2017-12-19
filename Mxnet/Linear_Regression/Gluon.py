# coding:utf-8

# coding:utf-8


from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon


class Gluon(object):
    def __init__(self):
        # 构造测试数据
        self.__num_inputs = None
        self.__num_examples = None
        self.__true_w = None
        self.__true_b = None
        self.__X = None
        self.__y = None

        # function set
        self.__net = None

        # goodness of function loss function
        self.__loss_function = None

        # goodness of function optimizer data
        self.__batch_size = None
        self.__data_iter = None

        # goodness of function optimizer function
        self.__trainer = None
        self.__learning_rate = None

        # pick the best function 模型训练
        self.__epochs = None
        self.__batch_X = None
        self.__batch_y = None
        self.__batch_y_hat = None

    def data_prepare(self):
        self.__num_inputs = 2
        self.__num_examples = 1000
        self.__true_w = nd.array([2, -3.4]).reshape((2, 1))
        self.__true_b = nd.array([4.2]).reshape((1, 1))
        self.__X = nd.random_normal(shape=(self.__num_examples, self.__num_inputs))
        self.__y = nd.dot(self.__X, self.__true_w) + self.__true_b
        self.__y += 0.001 * nd.random_normal(shape=self.__y.shape)

    def function_set(self):
        self.__net = gluon.nn.Sequential()
        self.__net.add(gluon.nn.Dense(1))
        self.__net.initialize()

    def goodness_of_function_loss_function(self):
        self.__loss_function = gluon.loss.L2Loss()

    def goodness_of_function_optimizer_data(self):
        self.__batch_size = 100
        self.__data_iter = (gluon.data.DataLoader(
            gluon.data.ArrayDataset(self.__X, self.__y), self.__batch_size, shuffle=True))

    def goodness_of_function_optimizer_function(self):
        self.__learning_rate = 1
        self.__trainer = gluon.Trainer(self.__net.collect_params(), "sgd", {"learning_rate": self.__learning_rate})

    def train_model(self):
        self.__epochs = 5
        for e in range(self.__epochs):
            total_loss = 0
            for self.__batch_X, self.__batch_y in self.__data_iter:
                with autograd.record():
                    self.__batch_y_hat = self.__net(self.__batch_X)
                    loss = self.__loss_function(self.__batch_y_hat, self.__batch_y)
                loss.backward()
                self.__trainer.step(self.__batch_size)
                total_loss += nd.sum(loss).asscalar()
            print("Epoch %d, average loss: %f" % (e, total_loss))


if __name__ == "__main__":
    s = Gluon()
    s.data_prepare()
    s.function_set()
    s.goodness_of_function_loss_function()
    s.goodness_of_function_optimizer_data()
    s.goodness_of_function_optimizer_function()
    s.train_model()