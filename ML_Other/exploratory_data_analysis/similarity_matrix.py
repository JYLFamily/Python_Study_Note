import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# 读入数据
def fread(path):
    raw_data = pd.read_csv(path, header=None, sep="\t")
    return raw_data

# 过滤方差 < 0.0001 的特征
def filter_approach_nonzero_features(raw_data, threshold=0.0001):
    apply_id_no = raw_data.iloc[:, 0]
    features = raw_data.iloc[:, 1:]

    selector = VarianceThreshold(threshold)
    selector.fit(features)

    return_data = pd.concat([pd.DataFrame(apply_id_no), pd.DataFrame(selector.fit_transform(features))], axis=1)
    return_data.columns = list(np.arange(return_data.shape[1]))

    return return_data

# 特征 k-means 聚类 , 如果一个簇中特征数 > 1 , 对这个簇的特征使用 PCA 降维只保留第一主成分
def features_k_means(raw_data, K=5):
    apply_id_no = raw_data.iloc[:, 0]
    # 特征 k-means , 对原始数据集转置即可
    features = raw_data.iloc[:, 1:].values.T

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(features_scaled)
    # 每个特征所属簇的 array
    clusters = kmeans.predict(features_scaled)
    print(clusters)
    temp = pd.DataFrame(apply_id_no)
    for k in np.arange(K):
        # 属于一个簇的特征 , PCA 之前矩阵转置回 (sample, features)
        features_k = features[list(clusters == k), :].T
        # 属于一个簇的特征数超过 1 , PCA 且只保留一个主成分
        if features_k.shape[1] > 1:
            pca = PCA(n_components=1)
            features_k = pca.fit_transform(features_k)

        # scaler = StandardScaler()
        # features_k_scaled = scaler.fit_transform(features_k)

        temp = pd.concat([temp, pd.DataFrame(features_k)], axis=1)

    temp.columns = list(np.arange(temp.shape[1]))

    return temp

# 绘制相似度矩阵图
def similarity_matrix(raw_data, columns_list=None):
    raw_data = raw_data.loc[:, [column for column in raw_data.columns if column in list(columns_list)]]
    corr_matrix = raw_data.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, vmax=0.8, square=True)
    sns.plt.show()

if __name__ == "__main__":
  path = ""
  train = fread(os.path.join(path, ""))
  train = filter_approach_nonzero_features(train)
  # train = features_k_means(train)
  # similarity_matrix(train, [1, 2, 3, 4, 5])
  # train.to_csv(path + "", header=False, sep="\t", index=False)

