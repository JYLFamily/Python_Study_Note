# coding:utf-8

import numpy as np
import pandas as pd


class CountResult(object):

    def __init__(self, input_path, sep, use_cols, percentile_list):
        self.__df = pd.read_csv(input_path, sep=sep, header=None, usecols=use_cols)
        self.__percentile_list = percentile_list
        self.__percentile_value = []

    def output_result(self):
        self.__df = self.__df.sort_values(by=[3], ascending=False)

        for i in self.__percentile_list:
            self.__percentile_value.append(np.percentile(self.__df[3], i))

        for i, j in zip(self.__percentile_value, self.__percentile_list):
            temp = self.__df.loc[(self.__df[3] > i), [2, 3]]
            print(i, end="    ")
            print(100 - j, end="    ")
            print("%.4f" % (temp.loc[(temp[2] == 1), :].shape[0]/float(temp.shape[0])))
        print("%.4f" % (self.__df.loc[(self.__df[2] == 1), :].shape[0]/float(self.__df.shape[0])))


if __name__ == "__main__":
    cr = CountResult("C:\\Users\\Dell\\Desktop\\result1.txt", sep="\t",
                     use_cols=[1, 2, 3], percentile_list=[95, 90, 85, 80, 75, 70])
    cr.output_result()