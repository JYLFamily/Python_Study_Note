# coding:utf-8

from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

class XgbSklearnTry(object,):

    def __init__(self, input_path, max_depth, learning_rate, num_round, silent, objective):
        self.__train, self.__train_label = load_svmlight_file(input_path + 'agaricus.txt.train')
        self.__test, self.__test_label = load_svmlight_file(input_path + 'agaricus.txt.test')
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

    def train_model(self):
        self.__bst = XGBClassifier(self.__max_depth
                                 , self.__learning_rate
                                 , self.__n_estimators
                                 , self.__silent
                                 , self.__objective)

        self.__bst.fit(self.__train, self.__train_label)

    def evaluate_train(self):
        self.__train_preds = self.__bst.predict(self.__train)
        self.__train_predictions = [round(value) for value in self.__train_preds]
        print("Train Accuary : %.2f%%" % (accuracy_score(self.__train_label, self.__train_predictions) * 100))

    def evaluate_test(self):
        self.__test_preds = self.__bst.predict(self.__test)
        self.__test_predictions = [round(value) for value in self.__test_preds]
        print("Test Accuary : %.2f%%" % (accuracy_score(self.__test_label, self.__test_predictions) * 100))


if __name__ == "__main__":
    xgb_sk_try = XgbSklearnTry(input_path='C:\\Users\\YL\\Desktop\\'
                               , max_depth=2
                               , learning_rate=1
                               , num_round=2
                               , silent=False
                               , objective='binary:logistic')

    xgb_sk_try.train_model()
    xgb_sk_try.evaluate_train()
    xgb_sk_try.evaluate_test()
