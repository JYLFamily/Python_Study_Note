# coding utf-8

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dropout, Dense, GRU, Embedding, Flatten, concatenate
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K


class Main(object):
    """ 程序正常运行
    """
    def __init__(self, *, input_path, batch_size, epochs):
        # data prepare
        self.__input_path = input_path
        self.__train, self.__train_label = [None for _ in range(2)]
        self.__keras_train = {}
        self.__keras_test = {}

        # function set & goodness of function
        self.__net = None

        # pick the best function
        self.__epochs = epochs
        self.__batch_size = batch_size

    def data_prepare(self):
        # 载入数据
        self.__train = pd.read_csv(
            os.path.join(self.__input_path, "train.tsv"),
            sep="\t",
            usecols=["train_id", "name", "item_condition_id",
                     "category_name", "brand_name", "shipping",
                     "item_description"]
        )
        self.__train_label = pd.read_csv(
            os.path.join(self.__input_path, "train.tsv"), sep="\t", usecols=["price"])

        # 缺失值填补
        self.__train["category_name"] = self.__train["category_name"].fillna("missing")
        self.__train["brand_name"] = self.__train["brand_name"].fillna("missing")
        self.__train["item_description"] = self.__train["item_description"].fillna("missing")

        # 分类变量 LabelEncoder
        self.__train["category_name"] = pd.Series(LabelEncoder().fit_transform(self.__train["category_name"]))
        self.__train["brand_name"] = pd.Series(LabelEncoder().fit_transform(self.__train["brand_name"]))

        # 文本变量分词并转换为序列
        tok = Tokenizer()
        tok.fit_on_texts(self.__train["name"].apply(lambda x: str(x).lower()))
        self.__train["seq_name"] = pd.Series(tok.texts_to_sequences(self.__train["name"].str.lower()))
        tok.fit_on_texts(self.__train["item_description"].apply(lambda x: str(x).lower()))
        self.__train["seq_item_description"] = \
            pd.Series(tok.texts_to_sequences(self.__train["item_description"].str.lower()))

        # 序列变量预处理并构成最终训练集 keras_train
        max_name_len = 10
        max_item_description_len = 75

        self.__keras_train["name"] = pad_sequences(self.__train["seq_name"], maxlen=max_name_len)
        self.__keras_train["item_description"] = \
            pad_sequences(self.__train["seq_item_description"], maxlen=max_item_description_len)
        self.__keras_train["brand_name"] = self.__train["brand_name"].values
        self.__keras_train["category_name"] = self.__train["category_name"].values
        self.__keras_train["item_condition_id"] = self.__train["item_condition_id"].values
        self.__keras_train["shipping"] = self.__train["shipping"].values

    def function_set(self):
        def rmsle_cust(y_true, y_pred):
            first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
            second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
            return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

        def get_net():
            # params
            dr_r = 0.05
            dr_d = 0.05
            max_text_len = np.max([
                np.max(self.__train["seq_name"].apply(lambda x: len(x))),
                np.max(self.__train["seq_item_description"].apply(lambda x: len(x)))
            ]) + 2
            max_brand = self.__train["brand_name"].max() + 1
            max_category_name = self.__train.category_name.max() + 1
            max_item_condition_id = self.__train["item_condition_id"].max() + 1

            # Inputs
            name = Input(shape=[self.__keras_train["name"].shape[1]], name="name")
            item_description = Input(shape=[self.__keras_train["item_description"].shape[1]], name="item_description")
            brand_name = Input(shape=[1], name="brand_name")
            category_name = Input(shape=[1], name="category_name")
            item_condition_id = Input(shape=[1], name="item_condition_id")
            shipping = Input(shape=[1], name="shipping")

            # Embeddings layers
            emb_name = Embedding(max_text_len, 50)(name)
            emb_item_desc = Embedding(max_text_len, 50)(item_description)
            emb_brand_name = Embedding(max_brand, 10)(brand_name)
            emb_category_name = Embedding(max_category_name, 10)(category_name)
            emb_item_condition = Embedding(max_item_condition_id, 5)(item_condition_id)

            # rnn layer
            rnn_layer1 = GRU(16)(emb_item_desc)
            rnn_layer2 = GRU(8)(emb_name)

            # main layer
            main_l = concatenate([
                Flatten()(emb_brand_name)
                , Flatten()(emb_category_name)
                , Flatten()(emb_item_condition)
                , rnn_layer1
                , rnn_layer2
                , shipping
            ])
            main_l = Dropout(dr_r)(Dense(128, activation='elu')(main_l))
            main_l = Dropout(dr_d)(Dense(64, activation='elu')(main_l))

            # output
            output = Dense(1, activation="linear")(main_l)

            # model
            model = Model(
                [name, item_description, brand_name, category_name, item_condition_id, shipping], output
            )
            model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])

            return model

        self.__net = get_net()

    def pick_the_best_function(self):
        self.__net.fit(self.__keras_train, self.__train_label.values,
                       batch_size=self.__batch_size, epochs=self.__epochs)


if __name__ == "__main__":
    m = Main(input_path="C:\\Users\\Dell\\Desktop", batch_size=20000, epochs=5)
    m.data_prepare()
    m.function_set()
    m.pick_the_best_function()