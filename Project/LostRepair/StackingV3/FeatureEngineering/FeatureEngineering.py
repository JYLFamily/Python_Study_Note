# coding:utf-8

import numpy as np
from Project.LostRepair.StackingV3.FeatureEngineering.FeaturePreProcessing import *
from Project.LostRepair.StackingV3.FeatureEngineering.FeatureGeneration import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class FeatureEngineering(object):

    @staticmethod
    def fe_linear_model(*, train, test, categorical_feature, numeric_feature):
        train_categorical = train[categorical_feature]
        train_numeric = train[numeric_feature]
        test_categorical = test[categorical_feature]
        test_numeric = test[numeric_feature]

        # 线性模型处理分类变量的缺失值
        # Pandas in Pandas out
        train_categorical, test_categorical = fpp_categorical_missing_value_linear_model(
            train_categorical,
            test_categorical
        )

        # 分类变量之间衍生变量
        # Pandas in Pandas out
        train_categorical, test_categorical = fg_categorical_categorical(
            train_categorical,
            test_categorical
        )

        # 分类变量与数值变量之间衍生变量
        # Pandas in Numpy out (可以输出 Pandas)
        train_numeric, test_numeric = fg_categorical_numeric(
            train_categorical,
            train_numeric,
            test_categorical,
            test_numeric
        )

        # 线性模型分类变量预处理
        # Pandas in Numpy out
        train_categorical, test_categorical, train_numeric_distribution, test_numeric_distribution = fpp_categorical_linear_model(
            train_categorical,
            test_categorical
        )

        # 线性模型处理数值变量的缺失值
        # Numpy in Numpy out
        train_numeric, test_numeric = fpp_numeric_missing_value_linear_model(
            np.hstack((train_numeric, train_numeric_distribution)),
            np.hstack((test_numeric, test_numeric_distribution))
        )

        # 线性模型数值变量预处理
        # Numpy in Numpy out
        train_numeric, test_numeric = fpp_numeric_linear_model(train_numeric, test_numeric)

        # 合并输出
        train_new = np.hstack((train_categorical, train_numeric))
        test_new = np.hstack((test_categorical, test_numeric))

        return train_new, test_new

    @staticmethod
    def fe_re_forest():
        pass

    @staticmethod
    def fe_cat_boosting():
        pass

    @staticmethod
    def fe_light_gbm():
        pass


if __name__ == "__main__":
    X = pd.read_csv(
        "D:\\Project\\LostRepair\\more_than_one_number\\train.csv",
        usecols=list(range(0, 4))
    )
    y = pd.read_csv(
        "D:\\Project\\LostRepair\\more_than_one_number\\train.csv",
        usecols=[4]
    )
    print(type(y.notnull()))
    X = X.loc[y.notnull().squeeze(), :]
    y = y.loc[y.notnull().squeeze(), :]

    train, test, train_label, test_label = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=9)
    train_new, test_new = FeatureEngineering.fe_linear_model(
        train=train,
        test=test,
        categorical_feature=["isSameLocation", "phoneCallLocation"],
        numeric_feature=["phoneCallTimesRation", "workTimeRatio"]
    )

    lr = LogisticRegression(C=0.5).fit(train_new, train_label)
    print(roc_auc_score(train_label, lr.predict_proba(train_new)[:, 1]))
    print(roc_auc_score(test_label, lr.predict_proba(test_new)[:, 1]))
