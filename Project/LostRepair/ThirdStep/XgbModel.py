# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 12, 4


class XgbModel(object):

    def __init__(self, train, train_label, test, test_label):
        self.__train = train
        self.__train_label = train_label
        self.__test = test
        self.__test_label = test_label
        self.__bst = None
        self.__feat_imp = None
        self.__test_preds = None
        self.__test_predictions = None
        self.__output = None

    def train(self):
        self.__bst = XGBClassifier(objective="binary:logistic")
        self.__bst.fit(self.__train, self.__train_label, eval_metric="auc")

    def predict(self):
        self.__test_preds = self.__bst.predict_proba(self.__test)[:, 1]
        self.__test_predictions = self.__bst.predict(self.__test)

    def feature_importances(self):
        self.__feat_imp = (pd.Series(self.__bst.feature_importances_, ["gbc", "rf", "ab", "lr"])
                           .sort_values(ascending=False))
        self.__feat_imp.plot(kind="bar", title="Feature Importances")
        plt.ylabel("Feature Importance Score")
        plt.show()

    def evaluate(self):
        print("auc : %.4f" % roc_auc_score(self.__test_label, self.__test_preds))
        print("accuracy score : %.4f" % accuracy_score(self.__test_label, self.__test_predictions))

    def evaluate_output(self):
        self.__output = np.hstack((self.__test, self.__test_label.reshape((-1, 1)), self.__test_preds.reshape((-1, 1))))
        print(self.__output.shape)
        pd.DataFrame(self.__output).to_csv("C:\\Users\\Dell\\Desktop\\output.csv")

    def xgbmodel_output(self):
        joblib.dump(self.__bst, "C:\\Users\\Dell\\Desktop\\bst.pkl.z", compress=3)