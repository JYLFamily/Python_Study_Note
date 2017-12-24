import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from numpy import float64
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# 读入数据
def fread(path):
    raw_data = pd.read_csv(path, sep="\t")
    return raw_data


# 过滤方差 < 0.0001 的特征
def filter_approach_nonzero_features(raw_data, threshold=0.0001):
    primary_key = raw_data.iloc[:, 0].to_frame()
    features = raw_data.iloc[:, 1:]

    primary_key_columns = primary_key.columns
    features_columns = features.columns

    selector = VarianceThreshold(threshold)
    selector.fit(features)

    return_raw_data = pd.concat([primary_key, \
                                 pd.DataFrame(selector.fit_transform(features),
                                              columns=features_columns[selector.get_support()])], axis=1)

    return return_raw_data


# 特征 k-means 聚类
def features_k_means(raw_data):
    raw_data = raw_data.drop([], axis=1)
    primary_key = raw_data.iloc[:, 0].to_frame()
    features = raw_data.iloc[:, 1:]

    primary_key_columns = primary_key.columns
    features_columns = features.columns

    features = features.values.T
    features = features.astype(float64)
    features = StandardScaler().fit_transform(features)

    for K in range(2, 10):
        # kmeans = KMeans(n_clusters=K, random_state=0).fit(features)
        # print("------------------------" + str(K) + "------------------------")
        # for k in range(K):
        #     print("cluster: " + str(k))
        #     print(features_columns[list(kmeans.labels_ == k)])
        #     print()
        # print(silhouette_score(features, kmeans.labels_, metric="euclidean"))
        kmeans = KMeans(n_clusters=K, n_jobs=-1, random_state=0).fit(features)
        print("------------------------" + str(K) + "------------------------")
        # print(kmeans.labels_)
        # print(list(features_columns))
        pd.DataFrame({"column":list(features_columns), "labels":kmeans.labels_}).\
            to_csv("C:\\Users\\Dell\\Desktop\\"+str(K), index=False)
        print(silhouette_score(features, kmeans.labels_, metric="euclidean"))
        print("-------------------------------------------------")


# 绘制相似度矩阵图
def similarity_matrix(raw_data):
    raw_data = raw_data.iloc[:, 1:]
    raw_data = raw_data.loc[:, []]
    corr_matrix = raw_data.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, vmax=0.8, square=True)
    sns.plt.show()


if __name__ == "__main__":
    path = ""
    train = fread(os.path.join(path, ""))
    features_k_means(train)

