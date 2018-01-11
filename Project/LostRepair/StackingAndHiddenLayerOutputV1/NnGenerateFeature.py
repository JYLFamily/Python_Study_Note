# coding:utf-8

import pandas as pd
import keras
from keras import backend as K
from keras import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from Project.LostRepair.StackingAndHiddenLayerOutputV1.FeatureEngineering import FeatureEngineering


class NnGenerateFeature(object):
    @staticmethod
    def get_intermediate_layer_output(*, train, train_label, test, test_label):
        if isinstance(train, pd.DataFrame):
            train = train.values

        if isinstance(train_label, pd.DataFrame):
            train_label = train_label.values

        if isinstance(test, pd.DataFrame):
            test = test.values

        if isinstance(test_label, pd.DataFrame):
            test_label = test_label.values

        # function set
        model = Sequential([
            # 4 层神经网络
            # 输入层 + 第一个隐藏层
            Dense(units=100, input_dim=train.shape[1]),
            Activation("relu"),
            # 第二个隐藏层
            Dense(units=100),
            Activation("relu"),
            # 输出层
            Dense(units=1),
            Activation("sigmoid")
        ])

        # goodness of function
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

        # pick the best function
        model.fit(train, train_label, epochs=5, batch_size=256)

        # get layer output
        intermediate_tensor_function = K.function([model.layers[0].input],
                                                  [model.layers[2].output])
        train_layer_output = intermediate_tensor_function([train])[0]
        test_layer_output = intermediate_tensor_function([test])[0]

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
