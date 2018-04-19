# coding:utf-8

import os
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score
from Project.LostRepair.StackingV3.Metric.Metrics import ar_ks_kendall_tau
from sklearn.externals import joblib


class YApplyBaseline(object):

    def __init__(self, *, input_path):
        self.__train_feature_woe = pd.read_csv(os.path.join(input_path, "train_feature_woe.csv"))
        self.__train_label = pd.read_csv(os.path.join(input_path, "train_label.csv"))
        self.__validation_feature_woe = pd.read_csv(os.path.join(input_path, "validation_feature_woe.csv"))
        self.__validation_label = pd.read_csv(os.path.join(input_path, "validation_label.csv"))
        self.__oot_feature_woe = pd.read_csv(os.path.join(input_path, "oot_feature_woe.csv"))
        self.__oot_label = pd.read_csv(os.path.join(input_path, "oot_label.csv"))

        self.__feature_columns = None

        self.__rus = None
        self.__train_us_feature_woe = None
        self.__train_us_label = None
        self.__train_us_validation_feature_woe = None
        self.__train_us_validation_label = None
        self.__train_us_validation_index = None

        self.__lr = None
        self.__ps = None
        self.__sfs = None
        self.__param = None
        self.__lr_bo = None

        self.__train_proba = None
        self.__train_score = None
        self.__validation_proba = None
        self.__validation_score = None
        self.__oot_proba = None
        self.__oot_score = None

        self.__oot_us_feature_woe = None
        self.__oot_us_label = None

    def type_transform(self):
        self.__feature_columns = self.__train_feature_woe.columns

        # Pandas to Numpy
        self.__train_feature_woe = self.__train_feature_woe.values
        self.__train_label = self.__train_label.squeeze().values
        self.__validation_feature_woe = self.__validation_feature_woe.values
        self.__validation_label = self.__validation_label.squeeze().values
        self.__oot_feature_woe = self.__oot_feature_woe.values
        self.__oot_label = self.__oot_label.squeeze().values

    def under_sampling(self):
        self.__rus = RandomUnderSampler(random_state=7)
        # Numpy in Numpy out
        self.__train_us_feature_woe, self.__train_us_label = self.__rus.fit_sample(self.__train_feature_woe, self.__train_label)
        self.__train_us_validation_feature_woe = np.vstack((self.__train_us_feature_woe, self.__validation_feature_woe))
        self.__train_us_validation_label = np.vstack((self.__train_us_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)))).reshape((-1, ))

        self.__train_us_validation_index = np.ones((self.__train_us_validation_label.shape[0], ))
        self.__train_us_validation_index[self.__train_us_label.shape[0]:] = -1

    def hyper_parameter_tunning(self):
        # 调整特征组合
        self.__lr = LogisticRegression()
        self.__ps = PredefinedSplit(self.__train_us_validation_index)
        self.__sfs = SequentialFeatureSelector(
            estimator=self.__lr,
            k_features=(1, 11),
            forward=True,
            floating=True,
            scoring="roc_auc",
            cv=self.__ps
        )
        self.__sfs.fit(self.__train_us_validation_feature_woe, self.__train_us_validation_label)
        # 最终模型使用的 feature
        self.__feature_columns = self.__feature_columns[list(self.__sfs.k_feature_idx_)]
        # Numpy 使用不同的方式进行索引 , OOT 已经不包含通过 SFFS 删掉的特征了
        self.__train_feature_woe = self.__train_feature_woe[:, self.__sfs.k_feature_idx_]
        self.__train_us_feature_woe = self.__train_us_feature_woe[:, self.__sfs.k_feature_idx_]
        self.__validation_feature_woe = self.__validation_feature_woe[:, self.__sfs.k_feature_idx_]

        self.__train_us_validation_feature_woe = self.__train_us_validation_feature_woe[:, self.__sfs.k_feature_idx_]

        # 特征组合一定的条件下调整 LR 超参数 C
        def __lr_cv(C):
            clf = LogisticRegression(
                C=C,
                random_state=7
            )
            val = cross_val_score(
                clf,
                self.__train_us_validation_feature_woe,
                self.__train_us_validation_label,
                scoring="roc_auc",
                cv=self.__ps
            ).mean()

            return val

        self.__param = {"C": (0.1, 100)}
        self.__lr_bo = BayesianOptimization(__lr_cv, self.__param, random_state=7)
        self.__lr_bo.maximize(** {"alpha": 1e-5})

    def fit_predict(self):
        self.__lr = LogisticRegression(C=round(self.__lr_bo.res["max"]["max_params"]["C"], 4))
        self.__lr.fit(self.__train_us_feature_woe,  self.__train_us_label)

        # self.__train_proba = pd.Series(self.__lr.predict_proba(self.__train_feature_woe)[:, 1].reshape((-1, )))
        # self.__train_score = self.__train_proba.apply(lambda x: 481.8621881 - 57.70780164 * np.log(x/(1-x)))
        # ar_ks_kendall_tau(self.__train_score.values, self.__train_label)
        #
        # self.__validation_proba = pd.Series(self.__lr.predict_proba(self.__validation_feature_woe)[:, 1].reshape((-1,)))
        # self.__validation_score = self.__validation_proba.apply(lambda x: 481.8621881 - 57.70780164 * np.log(x / (1 - x)))
        # ar_ks_kendall_tau(self.__validation_score.values, self.__validation_label)

        self.__oot_proba = pd.Series(self.__lr.predict_proba(self.__oot_feature_woe)[:, 1].reshape((-1,)))
        self.__oot_score = self.__oot_proba.apply(lambda x: 481.8621881 - 57.70780164 * np.log(x / (1 - x)))
        ar_ks_kendall_tau(self.__oot_score.values, self.__oot_label)

    def model_persistence(self):
        # self.__oot_us_feature_woe, self.__oot_us_label = self.__rus.fit_sample(self.__oot_feature_woe, self.__oot_label)
        # self.__oot_us_feature_woe = self.__oot_us_feature_woe
        # # feature oot_us_feature_woe 已经删掉了经过 SFFS 不要的特征 , 所以能够直接 vstack
        # feature = np.vstack((self.__train_us_validation_us_feature_woe, self.__oot_us_feature_woe))
        # label = np.vstack((self.__train_us_validation_us_label.reshape((-1, 1)), self.__oot_us_label.reshape((-1, 1)))).reshape((-1,))
        #
        # self.__lr.fit(feature, label)
        # print(self.__lr.coef_)
        joblib.dump(self.__lr, "C:\\Users\\Dell\\Desktop\\lr.z")


if __name__ == "__main__":
    yab = YApplyBaseline(input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\score_card\\yunyingshang\\2018-03-20\\data")
    yab.type_transform()
    yab.under_sampling()
    yab.hyper_parameter_tunning()
    yab.fit_predict()
    # yab.model_persistence()
