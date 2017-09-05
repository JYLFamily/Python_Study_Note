import numpy as np
import pandas as pd
from ML_Other.feature_engineering.io.read import read

# remove feature high correlation
class RFHC():
    def fit_transform(self, X, feature_index):
        """
        :param X: 数据集 , label + features
        :param feature_index: features 的下标 ( 下标指第几列 , 下面单词下标相同 )
        :return: 剔除相关系数 > 0.9 之后的数据集
        """
        X_labels = X.loc[:, [i for i in X.columns if i not in feature_index]]
        X_features = X.loc[:, feature_index]

        # corr 是 Array 类型
        # X_features 是 DataFrame 类型
        # corr 下标从 0 开始
        # X_features 下标从 1 开始
        # 故 corr 的下标应用到 X_features 的下标 , X_features 要 - 1
        corr = pd.DataFrame(X_features).corr().values
        corr = list(np.nonzero((corr > 0.9) & (corr != 1))[0])
        # 取 corr 列表偶数位置元素 [1::2] 奇数位置元素 ,
        # [16, 17, 18, 19, 20, 21]
        # (16, 17）
        # (18, 19)
        # (20, 21)
        # 以上是相关系数 > 0.9 的三组特征下标 , 剔除奇数位置元素、偶数位置元素均可
        corr = corr[::2]

        X_features =X_features.loc[:, [i for i in X_features.columns if i-(X.shape[1]-X_features.shape[1]) not in corr]]
        X = pd.concat([X_labels, X_features], axis=1, ignore_index=True).reset_index(drop=True)

        self.X = X

if __name__ == "__main__":
    X = read.fread(path="C:/Users/puhui/Desktop/2017/Auguest/credit_card.features.201607",
                   sep="\t",
                   header=None)
    rfhc = RFHC()
    rfhc.fit_transform(X, list(np.arange(1, 23, 1)))
    print(rfhc.X.corr() > 0.9)
    #print((X.loc[:, list(np.arange(1, 23, 1))].corr() > 0.9) & (X.loc[:, list(np.arange(1, 23, 1))].corr() != 1))
    #print((X.loc[:, [i for i in list(np.arange(1, 23, 1)) if i not in [17, 20, 21]]].corr() > 0.9) & \
    #      (X.loc[:, [i for i in list(np.arange(1, 23, 1)) if i not in [17, 20, 21]]].corr() != 1))