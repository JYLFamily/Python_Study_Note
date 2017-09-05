import numpy as np
import pandas as pd
from ML_Other.feature_engineering.io.read import read
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

class PFA():
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit_transform(self, X):
        if not self.q:
            self.q = X.shape[1]

        pca = PCA(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances(A_q[i, :], cluster_centers[c, :])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]