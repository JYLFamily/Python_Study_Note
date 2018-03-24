# coding:utf-8

import os
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import PredefinedSplit
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import Counter
from matplotlib import pyplot


class XgbBaselineV1(object):
    def __init__(self, *, input_path):
        self.__input_path = input_path
        self.__train_feature = pd.read_csv(os.path.join(self.__input_path, "train_feature.csv"), encoding="gbk")
        self.__train_label = pd.read_csv(os.path.join(self.__input_path, "train_label.csv"), encoding="gbk").squeeze().values
        self.__validation_feature = pd.read_csv(os.path.join(self.__input_path, "validation_feature.csv"), encoding="gbk")
        self.__validation_label = pd.read_csv(os.path.join(self.__input_path, "validation_label.csv"), encoding="gbk").squeeze().values

        self.__train_us_feature = None
        self.__train_us_label = None
        self.__train_us_validation_feature = None
        self.__train_us_validation_label = None
        self.__train_us_validation_index = None

        self.__params = None
        self.__xgb_bo = None
        self.__xgb = None

        self.__eval_set = []

    def set_feature(self):
        self.__train_feature = self.__train_feature.drop(["id_no", "apply_no", "create_date", "period", "real_amount", "status", "security_level"], axis=1)
        self.__validation_feature = self.__validation_feature.drop(["id_no", "apply_no", "create_date", "period", "real_amount", "status", "security_level"], axis=1)

        self.__train_feature = self.__train_feature.fillna(-999)
        self.__validation_feature = self.__validation_feature.fillna(-999)

        self.__train_feature = self.__train_feature.values
        self.__validation_feature = self.__validation_feature.values

        # rus = RandomUnderSampler()
        # Pandas in Numpy out
        # self.__train_us_feature, self.__train_us_label = rus.fit_sample(self.__train_feature, self.__train_label)

        self.__train_us_feature = self.__train_feature
        self.__train_us_label = self.__train_label

        self.__train_us_validation_feature = np.vstack((self.__train_us_feature, self.__validation_feature))
        self.__train_us_validation_label = np.vstack((self.__train_us_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)))).reshape((-1, ))
        self.__train_us_validation_index = np.ones((self.__train_us_validation_label.shape[0], ))
        self.__train_us_validation_index[self.__train_us_label.shape[0]:] = -1

    def fit_predict(self):
        # 得到超参数
        def __xgb_cv(max_depth, min_child_weight, learning_rate, n_estimators, subsample, colsample_bytree, colsample_bylevel):
            xgb = XGBClassifier(
                max_depth=int(max_depth),
                min_child_weight=min(min_child_weight, 1.0),

                learning_rate=min(learning_rate, 1.0),
                n_estimators=int(n_estimators),
                subsample=min(subsample, 1.0),
                colsample_bytree=min(colsample_bytree, 1.0),
                colsample_bylevel=min(colsample_bylevel, 1.0),

                objective="multi:softmax",
                missing=-999
            )
            val = cross_val_score(
                xgb,
                self.__train_us_validation_feature,
                self.__train_us_validation_label,
                scoring="f1_weighted",
                cv=PredefinedSplit(self.__train_us_validation_index)
            ).mean()

            return val

        self.__params = {"max_depth": (3, 10), "min_child_weight": (1 / np.sqrt(1000), 1 / np.sqrt(100)),
                         "learning_rate": (0.01, 1.0), "n_estimators": (5, 500),
                         "subsample": (0.3, 0.8), "colsample_bytree": (0.3, 0.8), "colsample_bylevel": (0.3, 0.8)}
        self.__xgb_bo = BayesianOptimization(__xgb_cv, self.__params, random_state=7)
        self.__xgb_bo.maximize(** {"alpha": 1e-5})
        print(self.__xgb_bo.res["max"]["max_params"])
        self.__xgb = XGBClassifier(
            max_depth=int(self.__xgb_bo.res["max"]["max_params"]["max_depth"]),
            min_child_weight=round(self.__xgb_bo.res["max"]["max_params"]["min_child_weight"], 4),

            learning_rate=round(self.__xgb_bo.res["max"]["max_params"]["learning_rate"], 4),
            n_estimators=int(self.__xgb_bo.res["max"]["max_params"]["n_estimators"]),
            subsample=round(self.__xgb_bo.res["max"]["max_params"]["subsample"], 4),
            colsample_bytree=round(self.__xgb_bo.res["max"]["max_params"]["colsample_bytree"], 4),
            colsample_bylevel=round(self.__xgb_bo.res["max"]["max_params"]["colsample_bylevel"], 4),

            objective="multi:softmax",
            missing=-999
        )

        # 如何训练
        self.__eval_set.append((self.__train_us_feature, self.__train_us_label))
        self.__eval_set.append((self.__validation_feature, self.__validation_label))
        self.__xgb.fit(self.__train_us_feature, self.__train_us_label, eval_metric=["mlogloss"], eval_set=self.__eval_set, early_stopping_rounds=20, verbose=True)

        # 训练图示
        results = self.__xgb.evals_result()
        epochs = len(results["validation_0"]["mlogloss"])
        x_axis = range(0, epochs)
        _, ax = pyplot.subplots()
        ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
        ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
        ax.legend()
        pyplot.xlabel("n estimators")
        pyplot.ylabel("m log loss")
        pyplot.title("Xgboost Log Loss")
        pyplot.show()

        print(accuracy_score(self.__validation_label, self.__xgb.predict(self.__validation_feature)))
        print(precision_score(self.__validation_label, self.__xgb.predict(self.__validation_feature), average=None))
        print(recall_score(self.__validation_label, self.__xgb.predict(self.__validation_feature), average=None))


if __name__ == "__main__":
    xbv = XgbBaselineV1(input_path="C:\\Users\\Dell\\Desktop\\week\\FC\\user_level\\data")
    xbv.set_feature()
    xbv.fit_predict()