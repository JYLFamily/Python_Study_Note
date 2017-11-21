# coding:utf-8

import pandas as pd
import matplotlib.pylab as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 12, 4


class SecondStepPlus(object):

    def __init__(self, input_path, sep, test_size, param=None, param_test=None, random_state=9, cv=10):
        self.__input_path = input_path
        self.__sep = sep
        self.__test_size = test_size
        self.__param = param
        self.__param_test = param_test
        self.__random_state = random_state
        self.__cv = cv
        self.__X = None
        self.__y = None
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__gbm = None
        self.__train_preds = None
        self.__train_predictions = None
        self.__test_preds = None
        self.__test_predictions = None
        self.__feat_imp = None

    def set_train_test(self):
        self.__X = pd.read_csv(self.__input_path, sep=self.__sep, usecols=list(range(1, 34)))
        self.__y = pd.read_csv(self.__input_path, sep=self.__sep, usecols=[0])
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

    def model_fit(self):
        self.__gbm = GradientBoostingClassifier(** self.__param, random_state=self.__random_state)
        self.__gbm.fit(self.__train, self.__train_label)

    def model_predict(self):
        # train set predict
        self.__train_preds = self.__gbm.predict(self.__train)
        self.__train_predictions = self.__gbm.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds = self.__gbm.predict(self.__test)
        self.__test_predictions = self.__gbm.predict_proba(self.__test)[:, 1]

    def model_evaluate(self):
        self.__feat_imp = pd.Series(self.__gbm.feature_importances_, self.__X.columns).sort_values(ascending=False)
        self.__feat_imp.plot(kind="bar", title="Feature Importances")
        plt.ylabel("Feature Importance Score")
        plt.show()

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds))


if __name__ == "__main__":
    param = {
        "learning_rate": 0.05,
        "n_estimators": 50,
        "subsample": 0.6,
        "max_depth": 7,
        "min_samples_split": 800,
        "min_samples_leaf": 50,
        "max_features": "sqrt"
    }

    ssp = SecondStepPlus(input_path="C:\\Users\\Dell\\Desktop\\zytsl_robot.csv",
                         sep=",",
                         test_size=0.2,
                         param=param)
    ssp.set_train_test()
    ssp.model_fit()
    ssp.model_predict()
    ssp.model_evaluate()