# encoding: utf-8
import os


def read_txt():
    # 返回一个文件对象
    f = open("C:\\Users\\Dell\\Desktop\\nohup.out")
    # 调用文件的 readline()方法
    line = f.readline()
    while line:
        # os.linesep 返回适合系统的换行符号 , 遇到空行跳过
        if line == os.linesep:
            continue
        if line.split(sep=" ")[0] == "cluster":
            print(line.split(sep=" ")[4], end="")
        line = f.readline()

    f.close()

if __name__ == "__main__":
    read_txt()