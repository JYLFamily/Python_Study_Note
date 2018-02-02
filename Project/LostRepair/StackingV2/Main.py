# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


class Main(object):

    def __init__(self, *, input_path, header, test_size, random_state):
        # data prepare
        self.__input_path = input_path
        self.__X = pd.read_csv(self.__input_path, header=header, usecols=list(range(0, 4))).values
        self.__y = pd.read_csv(self.__input_path, header=header, usecols=[4]).values.reshape((-1, ))
        self.__test_size, self.__random_state = test_size, random_state
        self.__train, self.__train_label, self.__test, self.__test_label = [None for _ in range(4)]

        # function set
        self.__lr = None
        self.__gb = None
        self.__rf = None
        self.__knn = None
        self.__nb = None
        self.__xgb = None
        self.__sclf = None

        # goodness_of_function
        self.__params = None
        self.__grid = None

    def data_prepare(self):
        self.__X = self.__X[np.logical_not(np.isnan(self.__y)), :]
        self.__y = self.__y[np.logical_not(np.isnan(self.__y))]

        self.__train, self.__test, self.__train_label, self.__test_label = train_test_split(
            self.__X,
            self.__y,
            test_size=self.__test_size,
            random_state=self.__random_state
        )

        scaler = StandardScaler().fit(self.__train)
        self.__train = scaler.transform(self.__train)
        self.__test = scaler.transform(self.__test)

    def function_set(self):
        self.__lr = LogisticRegression()
        self.__gb = GradientBoostingClassifier()
        self.__rf = RandomForestClassifier()
        self.__knn = KNeighborsClassifier()
        self.__nb = GaussianNB()
        self.__xgb = XGBClassifier()
        self.__sclf = StackingCVClassifier(
            classifiers=[self.__lr, self.__gb, self.__rf, self.__knn, self.__nb],
            meta_classifier=self.__xgb,
            use_probas=True,
            cv=5,
            use_features_in_secondary=True,
            verbose=1
        )

    def goodness_of_function(self):
        self.__params = {
            "logisticregression__C": [0.1, 0.3, 0.5, 0.8, 1, 3, 5, 8, 10]
            , "gradientboostingclassifier__learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5]
            , "gradientboostingclassifier__n_estimators": [50, 100, 150, 200]
            , "randomforestclassifier__n_estimators": [5, 10, 15, 20]
            , "kneighborsclassifier__n_neighbors": [3, 5, 8]
            , "meta-xgbclassifier__learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5]
            , "meta-xgbclassifier__n_estimators": [50, 100, 150, 200]
        }
        self.__grid = GridSearchCV(
            estimator=self.__sclf,
            param_grid=self.__params,
            scoring="roc_auc",
            n_jobs=4,
            cv=5,
            refit=True
        )

        self.__grid.fit(self.__train, self.__train_label)
        print("Best parameters: %s" % self.__grid.best_params_)
        print("Training Auc: %.4f" % self.__grid.best_score_)
        print("Testing Auc: %.4f" %
              roc_auc_score(self.__test_label, self.__grid.predict_proba(self.__test)[:, 1]))


if __name__ == "__main__":
    m = Main(
        input_path="D:\\Project\\LostRepair\\more_than_one_number\\train.csv",
        header=0,
        test_size=0.2,
        random_state=9
    )
    m.data_prepare()
    m.function_set()
    m.goodness_of_function()
