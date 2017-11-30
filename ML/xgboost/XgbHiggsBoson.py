# coding:utf-8

import pandas as pd


class XgbHiggsBoson(object):

    def __init__(self, input_path):
        self.__data = pd.read_csv(input_path)
        self.__train = self.__data[[header for header in self.__data.columns if header != "Label"]]
        self.__train_label = self.__data["Label"]
        print(self.__train.head())
        print(self.__train_label.head())


if __name__ == "__main__":
    xhb = XgbHiggsBoson("C:\\Users\\Dell\\Desktop\\higgsboson_training.csv")
