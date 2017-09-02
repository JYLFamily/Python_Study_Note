import os
import re
import numpy as np
import pandas as pd

def dir_files(path, patt):
    files = []
    pathDir = os.listdir(path)
    for allDir in pathDir:
        if allDir.startswith('.'):
            continue
        if re.search(patt, allDir) == None:
            continue
        child = os.path.join("%s/%s" % (path, allDir))
        files.append(child)
    return files

def load_data(path, sep, header):
    return pd.DataFrame(pd.read_table(path, sep=sep, header=header))


if __name__ == "__main__":
    data = load_data("C:/Users/puhui/Desktop/2017/Auguest/credit_card.features.201607", sep="\t", header=None)
    print(data.head())
