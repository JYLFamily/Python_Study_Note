# encoding: utf-8
import os


def read_txt():
    # try:
    #     # 返回一个文件对象
    #     f = open("C:\\Users\\Dell\\Desktop\\nohup.out", "r")
    #     # 调用文件的 readline()方法
    #     line = f.readline()
    #     while line:
    #         # os.linesep 返回适合系统的换行符号 , 遇到空行跳过
    #         if line == os.linesep:
    #             continue
    #         if line.split(sep=" ")[0] == "cluster":
    #             # print() 函数不换行
    #             print(line.split(sep=" ")[4], end="")
    #         line = f.readline()
    # finally:
    #     f.close()

    with open("C:\\Users\\Dell\\Desktop\\nohup.out", "r") as f:
        line = f.readline()
        while line:
            # os.linesep 返回适合系统的换行符号 , 遇到空行跳过
            if line == os.linesep:
                continue
            if line.split(sep=" ")[0] == "cluster":
                # print() 函数不换行
                print(line.split(sep=" ")[4], end="")
            line = f.readline()

if __name__ == "__main__":
    read_txt()