import numpy as np
import pandas as pd
from Other.feature_engineering.load.feature_engineering_load import *
from sklearn.feature_selection import *

def unsupervised_model_feature_selection(data, feature_index):

    X_label = data.loc[:, [i for i in data.columns if i not in feature_index]]
    X_features = data.loc[:, feature_index]

    selector = VarianceThreshold(threshold=0.001).fit(X_features)
    X_features = selector.transform(X_features)

    corr = pd.DataFrame(X_features).corr().values
    corr = list(np.nonzero((corr > 0.9) & (corr != 1))[0])
    corr = corr[0:int(len(corr)/2)]

    X_features = pd.DataFrame(X_features)
    X_features = X_features.loc[:, [i for i in X_features.columns if i not in corr]]
    print(corr)
    print(X_features.head())


if __name__ == "__main__":
    data = load_data("C:/Users/puhui/Desktop/2017/Auguest/credit_card.features.201607", sep="\t", header=None)
    unsupervised_model_feature_selection(data, data.columns[1:])