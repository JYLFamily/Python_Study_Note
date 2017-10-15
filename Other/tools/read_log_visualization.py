# encoding: utf-8
import os
import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def read_log(key_words):
    # try:
    #     # 返回 file-like object
    #     f = open("C:\\Users\\Dell\\Desktop\\nohup.out", "r")
    #     # 调用 file-like object 的 readline()方法
    #     line = f.readline()
    #     while line:
    #         # os.linesep 返回适合系统的换行符号
    #         # 遇到空行跳过
    #         if line == os.linesep:
    #             continue
    #         if line.split(sep=" ")[0] == "cluster":
    #             # end="" 函数不换行
    #             print(line.split(sep=" ")[4], end="")
    #         line = f.readline()
    # finally:
    #     f.close()
    """
    :param key_words: 分析日志关键词 , 与日志每行第一个单词匹配
    :return: [[], [], ……] 每一个列表元素存储一个统计量所有数据
    """
    key_words = key_words
    # [[]] * len(key_words) 不行 , 回导致 , 嵌套列表元素在内存中地址相同
    statistics = [[] for i in range(len(key_words))]

    with open("C:\\Users\\YL\\Desktop\\nohup.out", "r") as f:
        line = f.readline()
        while line:
            if line == os.linesep:
                continue
            # line 每行结束会带有换行符 , line.strip() 去掉每行结束的换行符
            line = line.strip()
            line_list = line.split(sep=" ")
            for word, statistic in zip(key_words, statistics):
                print(word)
                if line_list[0] == word:
                    # 正则表达式 \d+ 匹配一个及一个以上数字
                    # Python 中 \ 转义 , r'' 中不用考虑转义
                    statistic.append([float(i) for i in line_list if re.match(r'\d+', i)][0])
            line = f.readline()

    return statistics


def show_log(log, key_words):
    if len(log) > 1:
        log = pd.DataFrame(log).T
        log.columns = key_words
    else:
        log = pd.Series(log)

    plt.figure()
    log.plot()
    # 必须有这行否则不能显示图片
    plt.show()


if __name__ == "__main__":
    log = read_log(["dpc", "cluster"])
    print(log[0])
    print(log[1])
    show_log(log, ["dpc", "cluster"])