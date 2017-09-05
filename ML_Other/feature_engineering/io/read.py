import re
import os
import pandas as pd

class read():
    @staticmethod
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

    @staticmethod
    def fread(path, sep, header):
        return pd.DataFrame(pd.read_table(path, sep=sep, header=header))