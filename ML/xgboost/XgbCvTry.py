# coding:utf-8

import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class XgbCvTry(object):

    def __init__(self, input_path
                 , max_depth=None
                 , learning_rate=None
                 , num_round=None
                 , silent=None
                 , objective=None
                 , seed=None
                 , n_splits=None
                 , param_grid=None):
        self.__train, self.__train_label = load_svmlight_file(input_path + 'agaricus.txt.train')
        self.__test, self.__test_label = load_svmlight_file(input_path + 'agaricus.txt.test')
        self.__test_preds = None
        self.__test_predictions = None
        self.__bst = None
        self.__clf = None
        self.__max_depth = max_depth
        self.__learning_rate = learning_rate
        self.__n_estimators = num_round
        self.__silent = silent
        self.__objective = objective
        self.__param_grid = param_grid
        self.__seed = seed
        self.__n_splits = n_splits
        self.__kfold = None
        self.__cv_result = None

    def use_cvs(self):
        self.__bst = XGBClassifier(self.__max_depth
                                   , self.__learning_rate
                                   , self.__n_estimators
                                   , self.__silent
                                   , self.__objective)

        self.__kfold = StratifiedKFold(n_splits=self.__n_splits
                                       , random_state=self.__seed)

        # scoring=None 调用 XGBClassifier 的 score 方法
        # XGBClassifier 继承自 BaseEstimator score 方法为 accuracy_score
        self.__cv_result = cross_val_score(self.__bst
                                           , self.__train
                                           , self.__train_label
                                           , scoring="roc_auc"
                                           , cv=self.__kfold)

        # 得到一组参数在在训练集上经过 10 折交叉验证的结果
        print(self.__cv_result.mean(), self.__cv_result.std())

    def use_gcv(self):
        self.__bst = XGBClassifier()
        self.__clf = GridSearchCV(estimator=self.__bst
                                  , param_grid=self.__param_grid
                                  , cv=self.__n_splits
                                  , n_jobs=-1)
        # 这里一定要先 clf.fit()
        # 因为 refit=True 所以 clf.fit() 之后 clf.predict() 即可 , 与 clf.best_estimator_.predict() 一样
        self.__clf.fit(self.__train, self.__train_label)
        self.__test_preds = self.__clf.best_estimator_.predict(self.__test)
        self.__test_predictions = [round(value) for value in self.__test_preds]
        print("Test Accuary : %.2f%%" % (accuracy_score(self.__test_label, self.__test_predictions) * 100))


if __name__ == "__main__":
    # xgb_cv_try = XgbCvTry(input_path='D:\\Code\\Python\\Python_Study_Note\\ML\\xgboost\\'
    #                       , max_depth=2
    #                       , learning_rate=0.1
    #                       , num_round=2
    #                       , silent=True
    #                       , objective='binary:logistic'
    #                       , seed=9
    #                       , n_splits=10)
    # xgb_cv_try.use_cvs()

    param_grid = {
        'max_depth':[2, 3, 4],
        'learning_rate':[0.1, 0.3, 0.5, 0.7],
        'n_estimators':[2, 4, 6],
        'silent':[True],
        'objective':['binary:logistic']
    }

    xgb_cv_try = XgbCvTry(input_path='D:\\Code\\Python\\Python_Study_Note\\ML\\xgboost\\'
                          , param_grid = param_grid
                          , seed=9
                          , n_splits=10)
    xgb_cv_try.use_gcv()