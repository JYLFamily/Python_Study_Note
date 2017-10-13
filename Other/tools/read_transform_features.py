import pandas as pd
import numpy as np
import time
import datetime


def fread():
    all_01 = pd.read_csv("", sep="\t")
    print(all_01.shape)

    return all_01


def transform_features(raw_data):
    raw_data = raw_data.loc[:, []]

    print(raw_data.head())

if __name__ == "__main__":
    all_01 = fread()
    transform_features(all_01)
