import os
import pandas as pd
import numpy as np


def read_select_features(dir, file_name_list):
    for file_name in file_name_list:
        temp = pd.read_csv(os.path.join(dir, file_name), sep="\t", header=None)
        temp = temp.iloc[:, [0, 1, 11, 12, 13, 20]]
        temp.to_csv(os.path.join(dir, file_name) + "_new", sep="\t", header=False, index=False)
    print("finish!")


if __name__ == "__main__":
    read_select_features("C:\\Users\\Dell\\Desktop\\week", \
                         ["", \
                          "", \
                          ""])