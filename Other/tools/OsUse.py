# coding:utf-8

import os


class OsUse(object):

    def __init__(self, input_path, file_name):
        self.__input_path = input_path
        self.__file_name = file_name
        self.__dir_path = None
        self.__file_path = None

    def search_other(self):
        # self.__input_path.split(os.sep) 使用操作系统路径分隔符分割得到
        # In "C:\\Users\\Dell\\Desktop\\features_all.csv"
        # Out['C:', 'Users', 'Dell', 'Desktop', 'features_all.csv']

        # self.__input_path.split(os.sep)[:-1] 返回剔除最后一个元素 list 切片
        # In ['C:', 'Users', 'Dell', 'Desktop', 'features_all.csv']
        # Out['C:', 'Users', 'Dell', 'Desktop']

        # os.path.join(* self.__input_path.split(os.sep)[:-1]) list 解包使得能够使用 os.path.join() 函数
        self.__dir_path = os.path.join(* self.__input_path.split(os.sep)[:-1])
        print(self.__dir_path)

        # 得到新路径
        # C:Users\Dell\Desktop\RF.pkl.z
        self.__file_path = os.path.join(self.__dir_path, self.__file_name)
        print(self.__file_path)

if __name__ == "__main__":
    ou = OsUse("C:\\Users\\Dell\\Desktop\\features_all.csv", "RF.pkl.z")
