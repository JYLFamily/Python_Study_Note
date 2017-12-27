# coding:utf-8

import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


class ClassifierDemoPlus(object):

    def __init__(self):
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__model = None
        self.__clf = None
        self.__params = {"batch_size": [10, 50, 100],
                         "units": [10, 50, 100],
                         "activation": ["relu", "sigmoid"]}

    def train_test_split(self):
        (self.__train, self.__train_label), (self.__test, self.__test_label) = mnist.load_data()

        self.__train = self.__train.reshape((self.__train.shape[0], -1)) / float(255)
        self.__test = self.__test.reshape((self.__test.shape[0], -1)) / float(255)

        self.__train_label = to_categorical(self.__train_label, num_classes=10)
        self.__test_label = to_categorical(self.__test_label, num_classes=10)

    def train_model(self):
        def create_model(units, activation):
            model = Sequential([
                # 添加 "层" Dense(output_dim, input_dim)
                Dense(units, input_dim=784),
                # 添加 "激活函数"
                Activation(activation),
                Dense(units),
                Activation(activation),
                Dense(10),
                Activation("softmax")
            ])

            model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])

            return model

        self.__model = KerasClassifier(build_fn=create_model)
        self.__clf = GridSearchCV(self.__model,
                                  self.__params)
        self.__clf.fit(self.__train, self.__train_label)
        print(self.__clf.best_params_)

    def test_model(self):
        print(roc_auc_score(self.__test_label, self.__clf.predict_proba(self.__test)))


if __name__ == "__main__":
    cdp = ClassifierDemoPlus()
    cdp.train_test_split()
    cdp.train_model()
    # cdp.test_model()