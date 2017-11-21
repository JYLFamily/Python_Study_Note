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


class SecondStep(object):

    def __init__(self, input_path, sep=",", test_size=0.2, random_state=9, cv=10):
        self.__input_path = input_path
        self.__sep = sep
        self.__test_size = test_size
        self.__random_state = random_state
        self.__cv = cv
        self.__X = None
        self.__y = None
        self.__train = None
        self.__train_label = None
        self.__test = None
        self.__test_label = None
        self.__gbm_zero = None
        self.__gbm_first = None
        self.__gbm_second = None
        self.__gbm_third = None
        self.__gbm_fourth = None
        self.__gbm_fifth = None
        self.__gbm_sixth = None
        self.__gbm_seventh = None
        #
        self.__test_preds_zero = None
        self.__test_predictions_zero = None
        self.__train_preds_zero = None
        self.__train_predictions_zero = None
        #
        self.__test_preds_first = None
        self.__test_predictions_first = None
        self.__train_preds_first = None
        self.__train_predictions_first = None
        #
        self.__test_preds_second = None
        self.__test_predictions_second = None
        self.__train_preds_second = None
        self.__train_predictions_second = None
        #
        self.__test_preds_third = None
        self.__test_predictions_third = None
        self.__train_preds_third = None
        self.__train_predictions_third = None
        #
        self.__test_preds_fourth = None
        self.__test_predictions_fourth = None
        self.__train_preds_fourth = None
        self.__train_predictions_fourth = None
        #
        self.__test_preds_fifth = None
        self.__test_predictions_fifth = None
        self.__train_preds_fifth = None
        self.__train_predictions_fifth = None
        #
        self.__test_preds_sixth = None
        self.__test_predictions_sixth = None
        self.__train_preds_sixth = None
        self.__train_predictions_sixth = None
        #
        self.__test_preds_seventh = None
        self.__test_predictions_seventh = None
        self.__train_preds_seventh = None
        self.__train_predictions_seventh = None
        #
        self.__feat_imp_zero = None
        self.__feat_imp_seventh = None

    def set_train_test(self):
        self.__X = pd.read_csv(self.__input_path, sep=self.__sep, usecols=list(range(1, 34)))
        self.__y = pd.read_csv(self.__input_path, sep=self.__sep, usecols=[0])
        self.__train, self.__test, self.__train_label, self.__test_label = (
            train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_state))

    def model_fit_zero(self):
        """base line"""
        # fit
        self.__gbm_zero = GradientBoostingClassifier(random_state=self.__random_state)
        self.__gbm_zero.fit(self.__train, self.__train_label)
        # train set predict
        self.__train_preds_zero = self.__gbm_zero.predict(self.__train)
        self.__train_predictions_zero = self.__gbm_zero.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_zero = self.__gbm_zero.predict(self.__test)
        self.__test_predictions_zero = self.__gbm_zero.predict_proba(self.__test)[:, 1]

        # train set feature importance
        self.__feat_imp_zero = (pd.Series(self.__gbm_zero.feature_importances_, self.__X.columns)
                                .sort_values(ascending=False))
        self.__feat_imp_zero.plot(kind="bar", title="Feature Importances")
        plt.ylabel("Feature Importance Score")
        plt.show()

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_zero))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_zero))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_zero))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_zero))

    def model_fit_first(self):
        """fix learning_rate get n_estimators"""

        # boosting parameter
        param_test = {"n_estimators": range(20, 81, 10)}
        learning_rate = 0.1
        subsample = 0.8

        # CART Tree parameter
        min_samples_split = 750
        min_samples_leaf = 50
        max_depth = 8
        max_features = "sqrt"

        # fit
        self.__gbm_first = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=learning_rate
                                                 , subsample=subsample
                                                 , min_samples_split=min_samples_split
                                                 , min_samples_leaf=min_samples_leaf
                                                 , max_depth=max_depth
                                                 , max_features=max_features
                                                 , random_state=self.__random_state)
            , param_grid=param_test
            , scoring="roc_auc"
            , n_jobs=6
            , cv=self.__cv
            , verbose=True)
        self.__gbm_first.fit(self.__train, self.__train_label)

        # train set predict
        self.__train_preds_first = self.__gbm_first.predict(self.__train)
        self.__train_predictions_first = self.__gbm_first.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_first = self.__gbm_first.predict(self.__test)
        self.__test_predictions_first = self.__gbm_first.predict_proba(self.__test)[:, 1]

        # best n_estimators
        print(self.__gbm_first.best_params_, self.__gbm_first.best_score_)

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_first))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_first))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_first))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_first))

    def model_fit_second(self):
        """fix learning_rate and n_estimators get max_depth and num_samples_split"""

        # boosting parameter
        n_estimators = 50
        learning_rate = 0.05
        subsample = 0.8

        # CART Tree parameter
        param_test = {"max_depth": range(5, 16, 2),
                      "min_samples_split": range(200, 1001, 200)}
        min_samples_leaf = 50
        max_features = "sqrt"

        # fit
        self.__gbm_second = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=learning_rate
                                                 , subsample=subsample
                                                 , n_estimators=n_estimators
                                                 , min_samples_leaf=min_samples_leaf
                                                 , max_features=max_features
                                                 , random_state=self.__random_state)
            , param_grid=param_test
            , scoring="roc_auc"
            , n_jobs=6
            , cv=self.__cv
            , verbose=True)
        self.__gbm_second.fit(self.__train, self.__train_label)

        # train set predict
        self.__train_preds_second = self.__gbm_second.predict(self.__train)
        self.__train_predictions_second = self.__gbm_second.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_second = self.__gbm_second.predict(self.__test)
        self.__test_predictions_second = self.__gbm_second.predict_proba(self.__test)[:, 1]

        # best max_depth and min_samples_split
        print(self.__gbm_second.best_params_, self.__gbm_second.best_score_)

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_second))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_second))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_second))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_second))

    def model_fit_third(self):
        """fix learning_rate and n_estimators and max_depth and num_samples_split get min_samples_leaf"""

        # boosting parameter
        n_estimators = 50
        learning_rate = 0.05
        subsample = 0.8

        # CART Tree parameter
        param_test = {"min_samples_leaf":range(30, 71, 10)}
        max_depth = 7
        min_samples_split = 800
        max_features = "sqrt"

        # fit
        self.__gbm_third = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=learning_rate
                                                 , subsample=subsample
                                                 , n_estimators=n_estimators
                                                 , max_depth=max_depth
                                                 , min_samples_split=min_samples_split
                                                 , max_features=max_features
                                                 , random_state=self.__random_state)
            , param_grid=param_test
            , scoring="roc_auc"
            , n_jobs=6
            , cv=self.__cv
            , verbose=True)
        self.__gbm_third.fit(self.__train, self.__train_label)

        # train set predict
        self.__train_preds_third = self.__gbm_third.predict(self.__train)
        self.__train_predictions_third = self.__gbm_third.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_third = self.__gbm_third.predict(self.__test)
        self.__test_predictions_third = self.__gbm_third.predict_proba(self.__test)[:, 1]

        # best min_samples_leaf
        print(self.__gbm_third.best_params_, self.__gbm_third.best_score_)

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_third))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_third))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_third))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_third))

    def model_fit_fourth(self):
        """
        fix learning_rate
        and n_estimators
        and max_depth
        and num_samples_split
        and min_samples_leaf
        get max_features
        """

        # boosting parameter
        n_estimators = 50
        learning_rate = 0.05
        subsample = 0.8

        # CART Tree parameter
        param_test = {"max_features": range(1, 8, 2)}
        min_samples_leaf = 50
        max_depth = 7
        min_samples_split = 800

        # fit
        self.__gbm_fourth = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=learning_rate
                                                 , subsample=subsample
                                                 , n_estimators=n_estimators
                                                 , max_depth=max_depth
                                                 , min_samples_split=min_samples_split
                                                 , min_samples_leaf=min_samples_leaf
                                                 , random_state=self.__random_state)
            , param_grid=param_test
            , scoring="roc_auc"
            , n_jobs=6
            , cv=self.__cv
            , verbose=True)
        self.__gbm_fourth.fit(self.__train, self.__train_label)

        # train set predict
        self.__train_preds_fourth = self.__gbm_fourth.predict(self.__train)
        self.__train_predictions_fourth = self.__gbm_fourth.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_fourth = self.__gbm_fourth.predict(self.__test)
        self.__test_predictions_fourth = self.__gbm_fourth.predict_proba(self.__test)[:, 1]

        # best max_features
        print(self.__gbm_fourth.best_params_, self.__gbm_fourth.best_score_)

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_fourth))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_fourth))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_fourth))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_fourth))

    def model_fit_fifth(self):
        """
        fix learning_rate
        and n_estimators
        and max_depth
        and num_samples_split
        and min_samples_leaf
        and max_features
        get subsample
        """

        # boosting parameter

        param_test = {"subsample": [0.3, 0.4, 0.5, 0.6]}
        n_estimators = 50
        learning_rate = 0.05

        # CART Tree parameter
        min_samples_leaf = 50
        max_depth = 7
        min_samples_split = 800
        max_features = "sqrt"

        # fit
        self.__gbm_fifth = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=learning_rate
                                                 , n_estimators=n_estimators
                                                 , max_depth=max_depth
                                                 , min_samples_split=min_samples_split
                                                 , min_samples_leaf=min_samples_leaf
                                                 , max_features=max_features
                                                 , random_state=self.__random_state)
            , param_grid=param_test
            , scoring="roc_auc"
            , n_jobs=6
            , cv=self.__cv
            , verbose=True)
        self.__gbm_fifth.fit(self.__train, self.__train_label)
        # train set predict
        self.__train_preds_fifth = self.__gbm_fifth.predict(self.__train)
        self.__train_predictions_fifth = self.__gbm_fifth.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_fifth = self.__gbm_fifth.predict(self.__test)
        self.__test_predictions_fifth = self.__gbm_fifth.predict_proba(self.__test)[:, 1]

        # best max_depth
        print(self.__gbm_fifth.best_params_, self.__gbm_fifth.best_score_)

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_fifth))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_fifth))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_fifth))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_fifth))

    def model_fit_sixth(self):
        """
        and n_estimators
        and max_depth
        and num_samples_split
        and min_samples_leaf
        and max_features
        fix subsample
        reset learning_rate
        """

        # boosting parameter
        subsample = 0.6
        n_estimators = 50
        param_test = {"learning_rate": [0.0125, 0.025, 0.05]}

        # CART Tree parameter
        min_samples_leaf = 50
        max_depth = 7
        min_samples_split = 800
        max_features = "sqrt"

        # fit
        self.__gbm_sixth = GridSearchCV(
            estimator=GradientBoostingClassifier(n_estimators=n_estimators
                                                 , subsample=subsample
                                                 , max_depth=max_depth
                                                 , min_samples_split=min_samples_split
                                                 , min_samples_leaf=min_samples_leaf
                                                 , max_features=max_features
                                                 , random_state=self.__random_state)
            , param_grid=param_test
            , scoring="roc_auc"
            , n_jobs=6
            , cv=self.__cv
            , verbose=True)
        self.__gbm_sixth.fit(self.__train, self.__train_label)
        # train set predict
        self.__train_preds_sixth = self.__gbm_sixth.predict(self.__train)
        self.__train_predictions_sixth = self.__gbm_sixth.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_sixth = self.__gbm_sixth.predict(self.__test)
        self.__test_predictions_sixth = self.__gbm_sixth.predict_proba(self.__test)[:, 1]

        # best learning_rate
        print(self.__gbm_sixth.best_params_, self.__gbm_sixth.best_score_)

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_sixth))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_sixth))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_sixth))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_sixth))

    def model_fit_seventh(self):

        # boosting parameter
        n_estimators = 50
        learning_rate = 0.05
        subsample = 0.6

        # CART Tree parameter
        max_depth = 7
        min_samples_split = 800
        min_samples_leaf = 50
        max_features = "sqrt"

        self.__gbm_seventh = GradientBoostingClassifier(n_estimators=n_estimators
                                                        , learning_rate=learning_rate
                                                        , subsample=subsample
                                                        , max_depth=max_depth
                                                        , min_samples_split=min_samples_split
                                                        , min_samples_leaf=min_samples_leaf
                                                        , max_features=max_features
                                                        , random_state=self.__random_state)
        self.__gbm_seventh.fit(self.__train, self.__train_label)

        # train set predict
        self.__train_preds_seventh = self.__gbm_seventh.predict(self.__train)
        self.__train_predictions_seventh = self.__gbm_seventh.predict_proba(self.__train)[:, 1]
        # test set predict
        self.__test_preds_seventh = self.__gbm_seventh.predict(self.__test)
        self.__test_predictions_seventh = self.__gbm_seventh.predict_proba(self.__test)[:, 1]

        # train set feature importance
        self.__feat_imp_seventh = (pd.Series(self.__gbm_seventh.feature_importances_, self.__X.columns)
                                   .sort_values(ascending=False))
        self.__feat_imp_seventh.plot(kind="bar", title="Feature Importances")
        plt.ylabel("Feature Importance Score")
        plt.show()

        # train set AUC and Accuracy
        print("AUC(Train): %.4f" % roc_auc_score(self.__train_label, self.__train_predictions_seventh))
        print("Accuracy(Train): %.4f" % accuracy_score(self.__train_label, self.__train_preds_seventh))

        # test set AUC and Accuracy
        print("AUC(Test): %.4f" % roc_auc_score(self.__test_label, self.__test_predictions_seventh))
        print("Accuracy(Test): %.4f" % accuracy_score(self.__test_label, self.__test_preds_seventh))


if __name__ == "__main__":
    fs = SecondStep(input_path="C:\\Users\\Dell\\Desktop\\zytsl_robot.csv")
    fs.set_train_test()
    # fs.model_fit_zero()
    # fs.model_fit_first()
    # fs.model_fit_second()
    # fs.model_fit_third()
    # fs.model_fit_fourth()
    # fs.model_fit_fifth()
    # fs.model_fit_sixth()
    fs.model_fit_seventh()