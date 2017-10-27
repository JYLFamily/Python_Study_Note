# encoding: utf-8
import os
import re
import numpy as np
from numpy import double
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def read_log(key_words, re_exps):
    """
    :param key_words: 分析日志关键词 , 与日志每行第一个单词匹配
    :param re_exps: 匹配每个指标的正则表达式
    :return: [[], [], ……] 每一个列表元素存储一个统计量所有数据
    """
    key_words = key_words
    # [[]] * len(key_words) 不行 , 回导致 , 嵌套列表元素在内存中地址相同
    statistics = [[] for i in range(len(key_words))]

    with open("C:\\Users\\Dell\\Desktop\\phone_dpc_taobao.log", "r") as f:
        line = f.readline()
        while line:
            # 遇到空行跳过
            if line == os.linesep:
                continue
            # line 每行结束会带有换行符 , line.strip() 去掉每行结束的换行符
            line = line.strip()
            line_list = line.split(sep=" ")
            for word, re_exp, statistic in zip(key_words, re_exps, statistics):
                if line_list[0] == word:
                    # 正则表达式 \d+ 匹配一个及一个以上数字
                    # Python 中 \ 转义 , r'' 中不用考虑转义
                    statistic.append([double(re.findall(re_exp[1], i)[0]) \
                                      for i in line_list \
                                      if re.search(re_exp[0], i)])
            line = f.readline()

    return statistics


def compute_display_relative_entropy(raw_data):
    """
    :param raw_data:
    :return:
    """
    if type(raw_data) != pd.DataFrame:
       raw_data = pd.DataFrame(raw_data)

    kl_list = []
    for i in np.arange(1, raw_data.shape[0]):
        kl = 0
        for j in np.arange(0, raw_data.shape[1]):
            temp = 0
            p = raw_data.iloc[i, j]
            q = raw_data.iloc[i-1, j]

            if p == 0 and q == 0:
                temp = 0
            elif p == 0 and q != 0:
                temp = 0
            elif p != 0 and q == 0:
                temp = 10
            else:
                temp = p * np.log(p / q)
            kl = kl + temp

        # # print p 向量
        # print(raw_data.iloc[i, :])
        # print("kl : " + str(kl))

        kl_list.append(kl)

    plt.figure()
    pd.Series(kl_list).plot()
    plt.show()


def show_log(log, key_words = None):
    if len(log) > 1:
        log = pd.DataFrame(log).T
        log.columns = key_words
    else:
        log = pd.Series(log)

    plt.figure()
    log.plot()
    plt.show()


if __name__ == "__main__":
    log = read_log(["distribution_size", "distribution_amount"], \
                   [[r"\d+", r"\d+\.{1}\d+"], [r"\d+", r"\d+\.{1}\d+"]])
    # distribution_size
    compute_display_relative_entropy(log[0])
    # distribution_amount
    compute_display_relative_entropy(log[1])
