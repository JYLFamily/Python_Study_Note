import pandas as pd
import numpy as np
from numpy import float64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def fread(path):
    raw_data = pd.read_csv(path, sep="\t", header=None)
    raw_data = raw_data.values

    return  raw_data

def dist_eucl(sample_one, sample_two):
    """
    :param sample_one: 样本_1
    :param sample_two: 样本_2
    :return: 两样本的欧式距离
    """
    return np.sqrt(np.sum(np.power(sample_one - sample_two, 2)))

def choose_cluster_centroids(raw_data, K, sample_as_centroid=True):
    """
    :param raw_data:
    :param K:
    :param sample_as_centroid:
    :return:
    """

    if type(raw_data) == pd.DataFrame: raw_data = raw_data.values

    if sample_as_centroid:
        if type(raw_data) == np.ndarray: raw_data = pd.DataFrame(raw_data)
        centroids = raw_data.sample(n=K, replace=False).values
    else:
        # 特征数
        features_num = raw_data.shape[1]
        centroids = np.zeros((K, features_num))

        for f in range(features_num):
            min_f = raw_data[:, f].min()
            max_f = raw_data[:, f].max()
            range_f = max_f - min_f
            # np.random.rand(K) 生成 K 个 [0, 1] 之间的随机数 array
            centroids[:, f] = min_f + range_f * np.random.rand(K)

    return centroids

def k_means(raw_data, K=5, dist_func=dist_eucl, choose_func=choose_cluster_centroids, features_cluster=False):
    """
    :param raw_data:
    :param K:
    :param dist_func:
    :param choose_func:
    :return:
    """

    if type(raw_data) == pd.DataFrame: raw_data = raw_data.values

    id = raw_data[:, 0]
    if features_cluster:
        features = raw_data[:, 1:].T
    else:
        features = raw_data[:, 1:]

    features = features.astype(float64)
    features = StandardScaler().fit_transform(features)
    centroids = choose_func(features, K)

    # 样本数
    samples_num = features.shape[0]
    # 样本-簇-距离对应二维数组
    samples_centroid_map = np.zeros((samples_num, 2))

    samples_centroid_change = True
    i = 1
    while i < 30:
        samples_centroid_change = False

        print(centroids)

        # 更新样本所属聚类中心
        for sample_num in range(samples_num):
            min_index = -1
            min_dist = np.inf

            for centroid in range(K):
                dist = dist_func(features[sample_num, :], centroids[centroid, :])
                if dist < min_dist:
                    min_index = centroid
                    min_dist = dist

            if samples_centroid_map[sample_num, 0] != min_index: samples_centroid_change = True
            samples_centroid_map[sample_num, :] = min_index, min_dist ** 2

        # 更新簇聚类中心
        for centroid in range(K):
            cluster_k = features[list(samples_centroid_map[:, 0] == centroid), :]
            centroids[centroid, :] = cluster_k.mean(axis=0)

        i = i + 1

    return pd.concat([pd.DataFrame(id), pd.DataFrame(samples_centroid_map[:, 0])], axis=1).values


if __name__ == "__main__":
    train = fread("C:\\Users\\Dell\\Desktop\\week\\puhui_decision_existed_features.201701.json_features_taobao")
    print(k_means(train, 5))