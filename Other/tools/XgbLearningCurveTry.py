# coding:utf-8

from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


class XgbLearningCurveTry(object):

    def __init__(self, input_path, max_depth, learning_rate, num_round, silent, objective):
        self.__train, self.__train_label = load_svmlight_file(input_path + 'agaricus.txt.train')
        self.__test, self.__test_label = load_svmlight_file(input_path + 'agaricus.txt.test')
        self.__train_part = None
        self.__train_part_label = None
        self.__train_validation = None
        self.__train_validation_label = None
        self.__bst = None
        self.__max_depth = max_depth
        self.__learning_rate = learning_rate
        self.__n_estimators = num_round
        self.__silent = silent
        self.__objective = objective
        self.__train_preds = None
        self.__train_predictions = None
        self.__test_preds = None
        self.__test_predictions = None

    def use_validation(self, test_size, seed):
        self.__train_part, self.__train_validation, self.__train_part_label, self.__train_validation_label = (
            train_test_split(self.__train, self.__train_label, test_size=test_size, random_state=seed))

        eval_set = [(self.__train_part, self.__train_part_label)
            , (self.__train_validation, self.__train_validation_label)]

        self.__bst = XGBClassifier(self.__max_depth
                                   , self.__learning_rate
                                   , self.__n_estimators
                                   , self.__silent
                                   , self.__objective)

        self.__bst.fit(self.__train_part
                       , self.__train_part_label
                       , eval_metric=["error", "logloss"]
                       , eval_set=eval_set
                       , verbose=True)


if __name__ == "__main__":
    xgb_lc_try = XgbLearningCurveTry(input_path='C:\\Users\\YL\\Desktop\\'
                                     , max_depth=2
                                     , learning_rate=1
                                     , num_round=2
                                     , silent=False
                                     , objective='binary:logistic')

    xgb_lc_try.use_validation(test_size=0.33, seed=9)