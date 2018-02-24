# coding:utf-8

import numpy as np
from Project.LostRepair.StackingV3.FeatureEngineering.FeaturePreProcessing import *
from Project.LostRepair.StackingV3.FeatureEngineering.FeatureGeneration import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from Project.LostRepair.StackingV3.Metric.Metrics import ar_ks


class FeatureEngineering(object):

    @staticmethod
    def fe_linear_model(*, train, test, categorical_feature, numeric_feature):
        # 有分类变量与数值变量
        if (categorical_feature is not None) and (numeric_feature is not None):
            train_categorical = train[categorical_feature]
            test_categorical = test[categorical_feature]
            train_numeric = train[numeric_feature]
            test_numeric = test[numeric_feature]
        # 只有分类变量
        elif categorical_feature is not None:
            train_categorical = train
            test_categorical = test
        # 只有数值变量
        elif numeric_feature is not None:
            train_numeric = train
            test_numeric = test
        else:
            pass

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
        train_linear_model = np.hstack((train_categorical, train_numeric))
        test_linear_model = np.hstack((test_categorical, test_numeric))

        return train_linear_model, test_linear_model

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
    X = X.loc[y.notnull().squeeze(), :]
    y = y.loc[y.notnull().squeeze(), :]
    # array([0., 1.]), array([2525, 4228] 不能说是样本不平衡
    # print(np.unique(y.squeeze(), return_counts=True))

    train, test, train_label, test_label = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=9)
    train_label = train_label.values.reshape((-1, ))
    test_label = test_label.values.reshape((-1, ))
    train_linear_model, test_linear_model = FeatureEngineering.fe_linear_model(
        train=train,
        test=test,
        categorical_feature=["isSameLocation", "phoneCallLocation"],
        numeric_feature=["phoneCallTimesRation", "workTimeRatio"]
    )
    lr = LogisticRegression()
    lr.fit(train_linear_model, train_label)
    ar_ks(pd.Series(lr.predict_proba(test_linear_model)[:, 1]), pd.Series(test_label))
    # 以下实现的是一个简单的 lr "随机森林"版 , 能够缓解但是还是不能解决过拟合的问题
    # train_auc = {}
    # test_auc = {}
    # for i in range(15):
    #     # [0, train_linear_model.shape[0]) 中随机生成 size 个 int , 有重复
    #     sample = np.random.randint(0, train_linear_model.shape[0], size=int(train_linear_model.shape[0] * 0.6))
    #     col = np.random.randint(0, train_linear_model.shape[1], size=5)
    #     # train_linear_model[sample, :][:, col] 取子 Array
    #     # train_linear_model[[], []] 两个 list 必须等长 , 取出的元素行、列位置一一对应
    #     # C 还不是 lambda
    #     # Like in support vector machines, smaller values specify stronger regularization.
    #     lr = LogisticRegression().fit(train_linear_model[sample, :][:, col], train_label[sample])
    #     train_auc[str(i)] = lr.predict_proba(train_linear_model[:, col])[:, 1]
    #     test_auc[str(i)] = lr.predict_proba(test_linear_model[:, col])[:, 1]
    #
    # print(roc_auc_score(train_label, np.mean(pd.DataFrame(train_auc).values, axis=1)))
    # print(roc_auc_score(test_label, np.mean(pd.DataFrame(test_auc).values, axis=1)))

