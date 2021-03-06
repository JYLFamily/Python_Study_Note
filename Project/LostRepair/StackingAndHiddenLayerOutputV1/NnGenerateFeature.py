# coding:utf-8

import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from Project.LostRepair.StackingAndHiddenLayerOutputV1.FeatureEngineering import FeatureEngineering


class NnGenerateFeature(object):
    @staticmethod
    def get_intermediate_layer_output(*, train, train_label, test, test_label, cv, random_state):
        # function set
        model = Sequential([
            # 4 层神经网络
            # 输入层 + 第一个隐藏层
            Dense(units=100, input_dim=train.shape[1]),
            Activation("sigmoid"),
            # 第二个隐藏层
            Dense(units=100),
            Activation("sigmoid"),
            # 输出层
            Dense(units=1),
            Activation("sigmoid")
        ])

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        def get_oof_train(stacking_tuple):
            model, train, train_label = stacking_tuple

            # 100 是 第二个隐藏层 units 个数
            oof_train = np.zeros((train.shape[0], 100))

            for i, (train_index, test_index) in enumerate(skf.split(train, train_label)):
                x_train = train[train_index]
                y_train = train_label[train_index]
                x_test = train[test_index]

                # goodness of function
                model.compile(loss=keras.losses.binary_crossentropy,
                              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

                # pick the best function
                model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=False)

                # get layer output
                intermediate_tensor_function = K.function([model.layers[0].input],
                                                          [model.layers[2].output])

                oof_train[test_index] = intermediate_tensor_function([x_test])[0].reshape((-1, 100))

            return oof_train

        def get_oof_test(stacking_tuple):
            model, train, train_label, test = stacking_tuple

            # goodness of function
            model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

            # pick the best function
            model.fit(train, train_label, epochs=5, batch_size=256, verbose=False)

            # get layer output
            intermediate_tensor_function = K.function([model.layers[0].input],
                                                      [model.layers[2].output])

            oof_test = intermediate_tensor_function([test])[0].reshape((-1, 100))

            return oof_test

        train_layer_output = get_oof_train((model, train, train_label))
        test_layer_output = get_oof_test((model, train, train_label, test))

        return train_layer_output, test_layer_output


if __name__ == "__main__":
    X = pd.read_csv("C:\\Users\\Dell\\Desktop\\model.txt", header=None, sep="\t", usecols=list(range(1, 4)))
    y = pd.read_csv("C:\\Users\\Dell\\Desktop\\model.txt", header=None, sep="\t", usecols=[0])

    train_X, test_X, train_y, test_y = (
        train_test_split(X, y, test_size=0.2, random_state=9))

    train_X, test_X = (
        FeatureEngineering.net_model_feature_engineering(train=train_X,
                                                         test=test_X)
    )

    train_layer_output, test_layer_output = (
        NnGenerateFeature.get_intermediate_layer_output(train=train_X,
                                                        train_label=train_y,
                                                        test=test_X,
                                                        test_label=test_y)
    )
