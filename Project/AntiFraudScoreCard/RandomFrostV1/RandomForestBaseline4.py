# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import PredefinedSplit
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from Project.LostRepair.StackingV3.Metric.Metrics import ar_ks_kendall_tau
from sklearn.externals import joblib


class RandomForestBaseline4(object):
    def __init__(self, *, feature_header_list, label_header_list, tuning_method=True):
        # list
        self.__feature_header_list = feature_header_list
        # list
        self.__label_header_list = label_header_list
        # True PredefinedSplit False GridSearchCV
        self.__tuning_method = tuning_method

        self.__all = pd.read_csv(
            "C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data\\fc_qianzhan_product_anti_fraud_summary_loaned.csv",
            usecols=self.__feature_header_list+self.__label_header_list,
            encoding="gbk"
        )
        self.__oot = pd.read_csv(
            "C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data\\oot_loan_model.csv",
            usecols=self.__feature_header_list+self.__label_header_list,
            encoding="gbk"
        )
        self.__feature = None
        self.__label = None
        self.__oot_feature = None
        self.__oot_label = None
        self.__train_feature = None
        self.__train_label = None
        self.__validation_feature = None
        self.__validation_label = None
        self.__test_feature = None
        self.__test_label = None

        # sampling
        self.__train_us_feature = None
        self.__train_us_label = None
        self.__validation_us_feature = None
        self.__validation_us_label = None

        self.__train_us_validation_feature = None
        self.__train_us_validation_label = None
        self.__train_us_validation_index = None

        self.__train_us_validation_us_feature = None
        self.__train_us_validation_us_label = None

    def set_feature_label(self):
        self.__all = self.__all.loc[self.__all[self.__label_header_list].squeeze().notnull(), :]
        self.__oot = self.__oot.loc[self.__oot[self.__label_header_list].squeeze().notnull(), :]

        self.__feature = self.__all.loc[:, self.__feature_header_list]
        self.__feature = self.__feature.fillna(-999)
        self.__label = self.__all.loc[:, self.__label_header_list]

        self.__oot_feature = self.__oot.loc[:, self.__feature_header_list]
        self.__oot_feature = self.__oot_feature.fillna(-999)
        self.__oot_label = self.__oot.loc[:, self.__label_header_list]

        self.__train_feature, temp_feature, self.__train_label, temp_label = train_test_split(self.__feature, self.__label, test_size=0.4, shuffle=True)
        self.__validation_feature, self.__test_feature, self.__validation_label, self.__test_label = train_test_split(temp_feature, temp_label, test_size=0.5, shuffle=True)

        self.__train_feature = self.__train_feature.values
        self.__train_label = self.__train_label.squeeze().values
        self.__validation_feature = self.__validation_feature.values
        self.__validation_label = self.__validation_label.squeeze().values
        self.__test_feature = self.__test_feature.values
        self.__test_label = self.__test_label.squeeze().values
        self.__oot_feature = self.__oot_feature.values
        self.__oot_label = self.__oot_label.squeeze().values

    def sampling_fit_predict(self):
        if self.__tuning_method:
            rus = RandomUnderSampler()
            self.__train_us_feature, self.__train_us_label = rus.fit_sample(self.__train_feature, self.__train_label)

            self.__train_us_validation_feature = np.vstack((self.__train_us_feature, self.__validation_feature))
            self.__train_us_validation_label = np.vstack(
                (self.__train_us_label.reshape((-1, 1)), self.__validation_label.reshape((-1, 1)))).reshape((-1,))
            self.__train_us_validation_index = np.zeros((self.__train_us_validation_label.shape[0],))
            # 注意这里的问题
            self.__train_us_validation_index[self.__train_us_label.shape[0]:] = -1

            def __rf_cv(n_estimators, min_samples_split, max_features):
                pds = PredefinedSplit(self.__train_us_validation_index)
                val = cross_val_score(
                    RandomForestClassifier(
                        n_estimators=int(round(n_estimators, 1)),
                        min_samples_split=int(round(min_samples_split, 1)),
                        max_features=min(max_features, 1.0),
                        random_state=2
                    ),
                    self.__train_us_validation_feature,
                    self.__train_us_validation_label,
                    scoring="roc_auc",
                    cv=pds
                ).mean()
                return val

            rf_bo = BayesianOptimization(__rf_cv, {"n_estimators": (5, 250), "min_samples_split": (2, 25), "max_features": (0.1, 0.999)})
            rf_bo.maximize(**{"alpha": 1e-5})
            print(rf_bo.res["max"]["max_params"])
            rf = RandomForestClassifier(
                n_estimators=int(round(rf_bo.res["max"]["max_params"]["n_estimators"], 1)),
                min_samples_split=int(round(rf_bo.res["max"]["max_params"]["min_samples_split"], 1)),
                max_features=round(rf_bo.res["max"]["max_params"]["max_features"], 2)
            )
            rf.fit(self.__train_us_feature, self.__train_us_label)
            print("-" * 53)
            train_score = pd.Series(rf.predict_proba(self.__train_feature)[:, 1].reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x))).values
            ar_ks_kendall_tau(train_score.reshape(-1, ), self.__train_label.reshape(-1, ), loan=False)

            validation_score = pd.Series(rf.predict_proba(self.__validation_feature)[:, 1].reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x))).values
            ar_ks_kendall_tau(validation_score.reshape(-1, ), self.__validation_label.reshape(-1, ))

            test_score = pd.Series(rf.predict_proba(self.__test_feature)[:, 1].reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x))).values
            ar_ks_kendall_tau(test_score.reshape(-1, ), self.__test_label.reshape(-1, ))

            oot_score = pd.Series(rf.predict_proba(self.__oot_feature)[:, 1].reshape((-1,))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x / (1-x))).values
            ar_ks_kendall_tau(oot_score.reshape(-1, ), self.__oot_label.reshape(-1, ))
        else:
            rus = RandomUnderSampler()
            self.__train_us_feature, self.__train_us_label = rus.fit_sample(self.__train_feature, self.__train_label)
            self.__validation_us_feature, self.__validation_us_label = rus.fit_sample(self.__validation_feature, self.__validation_label)

            self.__train_us_validation_us_feature = np.vstack((self.__train_us_feature, self.__validation_us_feature))
            self.__train_us_validation_us_label = np.vstack(
                (self.__train_us_label.reshape((-1, 1)), self.__validation_us_label.reshape((-1, 1)))).reshape((-1,))

            def __rf_cv(n_estimators, min_samples_split, max_features):
                val = cross_val_score(
                    RandomForestClassifier(
                        n_estimators=int(round(n_estimators, 1)),
                        min_samples_split=int(round(min_samples_split, 1)),
                        max_features=min(max_features, 1.0),
                        random_state=2
                    ),
                    self.__train_us_validation_us_feature,
                    self.__train_us_validation_us_label,
                    scoring="roc_auc",
                    cv=3
                ).mean()
                return val

            rf_bo = BayesianOptimization(__rf_cv, {"n_estimators": (5, 250), "min_samples_split": (2, 25), "max_features": (0.1, 0.999)})
            rf_bo.maximize(init_points=5, n_iter=1, **{"alpha": 1e-5})
            print(rf_bo.res["max"]["max_params"])
            rf = RandomForestClassifier(
                n_estimators=int(round(rf_bo.res["max"]["max_params"]["n_estimators"], 1)),
                min_samples_split=int(round(rf_bo.res["max"]["max_params"]["min_samples_split"], 1)),
                max_features=round(rf_bo.res["max"]["max_params"]["max_features"], 2)
            )
            rf.fit(self.__train_us_feature, self.__train_us_label)
            print("-" * 53)
            train_score = pd.Series(rf.predict_proba(self.__train_feature)[:, 1].reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x))).values
            ar_ks_kendall_tau(train_score.reshape(-1, ), self.__train_label.reshape(-1, ))

            validation_score = pd.Series(rf.predict_proba(self.__validation_feature)[:, 1].reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x))).values
            ar_ks_kendall_tau(validation_score.reshape(-1, ), self.__validation_label.reshape(-1, ))

            test_score = pd.Series(rf.predict_proba(self.__test_feature)[:, 1].reshape((-1, ))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x))).values
            ar_ks_kendall_tau(test_score.reshape(-1, ), self.__test_label.reshape(-1, ))

            oot_score = pd.Series(rf.predict_proba(self.__oot_feature)[:, 1].reshape((-1,))).apply(lambda x: 481.8621881 - 28.85390082 * np.log(x / (1-x))).values
            ar_ks_kendall_tau(oot_score.reshape(-1, ), self.__oot_label.reshape(-1, ))


if __name__ == "__main__":
    feature_out = ["trans_num_max_by_date", "succ_trans_num_max_by_shop", "taobaoing_date", "open_last_days", "tb0020003", "tb0020010", "ep0010004", "mp0010009","ep0030009", "mp0010022", "cnt_call_std", "tb0020004", "ep0060003", "ep0060004"]
    label_out = ["if_fraud"]
    rfb = RandomForestBaseline4(feature_header_list=feature_out, label_header_list=label_out, tuning_method=False)
    rfb.set_feature_label()
    rfb.sampling_fit_predict()