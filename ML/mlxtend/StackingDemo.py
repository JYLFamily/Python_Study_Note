# coding:utf-8

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


class StackingDemo(object):
    def __init__(self):
        # data prepare
        self.__iris = None
        self.__X = None
        self.__y = None
        self.__train, self.__train_label = [None for _ in range(2)]
        self.__test, self.__test_label = [None for _ in range(2)]

        # function set
        self.__params = None

        self.__lr = None
        self.__gb = None
        self.__rf = None
        self.__sclf = None
        self.__grid = None

    def data_prepare(self):
        self.__iris = load_iris()
        self.__X = self.__iris.data[0:100]
        self.__y = self.__iris.target[0:100]
        self.__train, self.__test, self.__train_label, self.__test_label = train_test_split(
            self.__X,
            self.__y,
            test_size=0.2,
            shuffle=True
        )

    def function_set(self):
        # param
        self.__params = {
            # 注意名称必须是这样
            "logisticregression__C": list(np.linspace(start=0.1, stop=10, num=5)),
            "gradientboostingclassifier__learning_rate": list(np.linspace(start=0.1, stop=1, num=10)),
            "randomforestclassifier__n_estimators": list(range(5, 16)),
            "meta-logisticregression__C": list(np.linspace(start=0.1, stop=10, num=5))
        }

        # model
        self.__lr = LogisticRegression()
        self.__gb = GradientBoostingClassifier()
        self.__rf = RandomForestClassifier()
        self.__sclf = StackingCVClassifier(
            classifiers=[self.__lr, self.__gb, self.__rf],
            meta_classifier=self.__lr,
            use_probas=True,
            cv=5,
            use_features_in_secondary=True,
            verbose=1
        )

        self.__grid = GridSearchCV(
            estimator=self.__sclf,
            param_grid=self.__params,
            cv=5,
            refit=True
        )

    def goodness_of_function(self):
        self.__grid.fit(self.__train, self.__train_label)
        print("Best parameters: %s" % self.__grid.best_params_)
        print("Accuracy: %.2f" % self.__grid.best_score_)

    def pick_the_best_function(self):
        self.__lr = LogisticRegression(C=0.1)
        self.__gb = GradientBoostingClassifier(learning_rate=0.1)
        self.__rf = RandomForestClassifier(n_estimators=5)
        self.__sclf = StackingCVClassifier(
            classifiers=[self.__lr, self.__gb, self.__rf],
            meta_classifier=self.__lr,
            use_probas=True,
            cv=5,
            use_features_in_secondary=True,
            verbose=1
        )
        self.__sclf.fit(self.__train, self.__train_label)
        print(roc_auc_score(self.__test_label, self.__sclf.predict_proba(self.__test)[:, 1]))


if __name__ == "__main__":
    sd = StackingDemo()
    sd.data_prepare()
    sd.function_set()
    # sd.goodness_of_function()
    sd.pick_the_best_function()