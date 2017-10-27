# encoding: utf-8
import numpy as np
import time


def vectorized():
    # 设置随机数种子
    np.random.seed(1234)
    # np.random.rand(n, m) 生成 shape 为 (n, m) 的 array 每个元素从 [0, 1] 之间的均匀分布中抽取
    # 生成 Array 1000000 × 1 均匀分布随机数
    a = np.random.rand(1000000, 1)
    b = np.random.rand(1000000, 1)

    # 返回当前时间的 timestamp , timestamp 为当前时间距离 epoch time 的秒数
    tic = time.time()
    c = np.dot(a, b)
    toc = time.time()

    print(c)
    print("Vectorized version : " + str((toc - tic)) + " s")


def for_loop():
    # 设置随机数种子
    np.random.seed(1234)
    # 生成 Array 1000000 × 1 均匀分布随机数
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    tic = time.time()
    c = 0
    for i in np.arange(1000000):
        c += a[i] * b[i]
    toc = time.time()

    print(c)
    print("For loop version : " + str((toc - tic)) + " s")


if __name__ == "__main__":
    vectorized()
    for_loop()