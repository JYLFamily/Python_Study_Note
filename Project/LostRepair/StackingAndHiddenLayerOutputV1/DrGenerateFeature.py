# coding:utf-8

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DrGenerateFeature(object):
    @staticmethod
    def get_pca_component(*, train, test):
        pca = PCA(n_components=1).fit(train)
        train_component = pca.transform(train).reshape((-1, 1))
        test_component = pca.transform(test).reshape((-1, 1))

        return train_component, test_component

    @staticmethod
    def get_tsne_component(*, train, test):
        tsne = TSNE(n_components=1)
        train_component = tsne.fit_transform(train)
        test_component = tsne.fit_transform(test)

        return train_component, test_component


if __name__ == "__main__":
    train = np.arange(16).reshape((4, 4))
    test = np.arange(8).reshape((2, 4))
    train_pca_component, test_pca_component = DrGenerateFeature().get_pca_component(train=train, test=test)
    train_tsne_component, test_tsne_component = DrGenerateFeature().get_tsne_component(train=train, test=test)

    print(train_pca_component.shape)
    print(train_tsne_component.shape)
    print(train_pca_component)
    print(train_tsne_component)