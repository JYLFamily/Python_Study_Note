import pandas as pd
import numpy as np
from numpy import float64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def fread(path):
    raw_data = pd.read_csv(path, sep="\t", header=None)
    raw_data = raw_data.values
    return raw_data

def dist_eucl(sample_one, sample_two):
    return np.sqrt(np.sum(np.power(sample_one - sample_two, 2)))

def choose_centroid(raw_data, K, sample_as_centroid=True):
    if sample_as_centroid:
        if type(raw_data) == np.ndarray: raw_data = pd.DataFrame(raw_data)
        centroids = raw_data.sample(n=K, replace=False)
        centroids = centroids.values
    else:
        if type(raw_data) == pd.DataFrame: raw_data = raw_data.values
        feature_num = raw_data.shape[1]
        centroids = np.zeros((K, feature_num))
        for f in range(feature_num):
            min_f = raw_data[:, f].min()
            max_f = raw_data[:, f].max()
            range_f = max_f - min_f
            centroids[:, f] = min_f + range_f * np.random.rand(K, 1).T
    return centroids

def k_means(raw_data, K=5, dist_func=dist_eucl, choose_func=choose_centroid, features_cluster=False):
    # 分割数据集
    if type(raw_data) == pd.DataFrame:
        primary_key = raw_data.iloc[:, 0].values
        features = raw_data.iloc[:, 1:].values
    else:
        primary_key = raw_data[:, 0]
        features = raw_data[:, 1:]

    # 判断是特征聚类 , 还是样本聚类 , 特征聚类对原始数据集进行转置
    if features_cluster: features = features.T

    # 数据集标准化
    features = features.astype(float64)
    features_scaled = StandardScaler().fit_transform(features)

    # 选择聚类中心
    centroids = choose_func(features_scaled, K)

    # ---------------- 聚类 -----------------

    # ------ 变量 ------

    # 样本数
    sample_num = features_scaled.shape[0]
    # 样本-簇-距离映射二维数组
    sample_cluster_map = np.zeros((sample_num, 2))
    # 样本所属簇是否改变
    sample_cluster_change = True

    # ------ 循环 ------

    while sample_cluster_change:
        sample_cluster_change = False

        print(centroids)

        # ------ 更新 sample_cluster_map ------

        # 迭代样本
        for i in range(sample_num):
            # 到样本 i 距离最近的聚类中心
            min_index = -1
            # 到样本 i 距离最近的聚类中心的距离
            min_dist = np.inf
            # 计算样本距离每个聚类中心的距离 , 得到距离样本最近的聚类中心与样本距离该聚类中心的距离
            for j in range(K):
                dist = dist_func(features_scaled[i, :], centroids[j, :])
                # 聚类中心更新之后 , 样本只要找到样本距离当前聚类中心哪个最近即可 , 不必考虑上次聚类所属的聚类中心与所属聚类中心的距离
                if dist < min_dist:
                    min_index = j
                    min_dist = dist

            if sample_cluster_map[i, 0] != min_index: sample_cluster_change = True
            sample_cluster_map[i, 0] = min_index
            sample_cluster_map[i, 1] = min_dist ** 2

        # ------ 更新 centroids ------
        for j in range(K):
            sample_cluster_k = features_scaled[list(sample_cluster_map[:, 0] == j), :]
            centroids[j, :] = sample_cluster_k.mean(axis=0)


if __name__ == "__main__":
    train = fread("C:\\Users\\Dell\\Desktop\\week\\")
    k_means(train, 5)