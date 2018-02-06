 #coding:utf-8

import os
import zipfile


class LoadData(object):

    def __init__(self, *, zip_file_name, txt_file_name):
        self.__zip_file_name = zip_file_name
        self.__txt_file_name = txt_file_name
        self.__corpus_chars = None
        self.__char_to_idx = None
        self.__corpus_indices = None

    def set_data(self):
        with zipfile.ZipFile(self.__zip_file_name, "r") as zin:
            zin.extractall()
        with open(self.__txt_file_name, encoding="UTF-8") as f:
            self.__corpus_chars = f.read()
        # 所有系统的换行符都能够 replace
        self.__corpus_chars = self.__corpus_chars.replace("\r", " ").replace("\n", " ").replace(os.linesep, " ")

    def set_get_dict(self):
        """得到字典 返回字典"""
        # 字符去重
        idx_to_char = list(set(self.__corpus_chars))
        # "墟": 0 字典
        self.__char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

        return self.__char_to_idx

    def set_get_index(self):
        self.__corpus_indices = [self.__char_to_idx[char] for char in self.__corpus_chars]

        return self.__corpus_indices


if __name__ == "__main__":
    ld = LoadData(zip_file_name="jaychou_lyrics.zip", txt_file_name="jaychou_lyrics.txt")
    ld.set_data()
    ld.set_get_dict()
    ld.set_get_index()