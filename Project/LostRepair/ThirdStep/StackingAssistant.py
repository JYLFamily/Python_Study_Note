# coding:utf-8

from Project.LostRepair.ThirdStep.ModelDataAgg import ModelDataAgg
from Project.LostRepair.ThirdStep.OutOfFold import OutOfFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

from Project.LostRepair.ThirdStep.ModelAssistant import ModelAssistant


class StackingAssistant(object):

    def __init__(self, train, train_label, test,
                 lr_params=None, knn_params=None, extratree_params=None, adaboost_params=None, gbdt_params=None,
                 xgboost_params=None, randomforest_params=None, cv=2, random_state=9):
        self.__train = train
        self.__train_label = train_label
        self.__test = test
        self.__cv = cv
        self.__random_state = random_state
        # estimator , stage 1 使用的模型初始化
        self.__lr_estimator = ModelAssistant(clf=LogisticRegression, params=lr_params, random_state=self.__random_state)
        self.__knn_estimator = ModelAssistant(clf=KNeighborsClassifier, params=knn_params,
                                              random_state=self.__random_state)
        self.__et_estimator = ModelAssistant(clf=ExtraTreeClassifier, params=extratree_params,
                                             random_state=self.__random_state)
        self.__ad_estimator = ModelAssistant(clf=AdaBoostClassifier, params=adaboost_params,
                                             random_state=self.__random_state)
        self.__gbdt_estimator = ModelAssistant(clf=GradientBoostingClassifier, params=gbdt_params,
                                               random_state=self.__random_state)
        self.__xgb_estimator = ModelAssistant(clf=XGBClassifier, params=xgboost_params,
                                              random_state=self.__random_state)
        self.__rf_estimator = ModelAssistant(clf=RandomForestClassifier, params=randomforest_params,
                                             random_state=self.__random_state)
        #
        self.__lr_oof = None
        self.__lr_train_oof = None
        self.__lr_test_oof = None
        #
        self.__knn_oof = None
        self.__knn_train_oof = None
        self.__knn_test_oof = None
        #
        self.__et_oof = None
        self.__et_train_oof = None
        self.__et_test_oof = None
        #
        self.__ad_oof = None
        self.__ad_train_oof = None
        self.__ad_test_oof = None
        #
        self.__gbdt_oof = None
        self.__gbdt_train_oof = None
        self.__gbdt_test_oof = None
        #
        self.__xgb_oof = None
        self.__xgb_train_oof = None
        self.__xgb_test_oof = None
        #
        self.__rf_oof = None
        self.__rf_train_oof = None
        self.__rf_test_oof = None
        #
        self.__train_all = None
        self.__test_all = None

    def calc_feature(self):
        self.__lr_oof = OutOfFold(clf=self.__lr_estimator, train=self.__train, train_label=self.__train_label,
                                  test=self.__test, cv=self.__cv, random_state=self.__random_state)
        self.__lr_oof.set_skf()
        self.__lr_train_oof, self.__lr_test_oof = self.__lr_oof.get_oof()
        print("--- lr ready ! ---")

        self.__knn_oof = OutOfFold(clf=self.__knn_estimator, train=self.__train, train_label=self.__train_label,
                                   test=self.__test, cv=self.__cv, random_state=self.__random_state)
        self.__knn_oof.set_skf()
        self.__knn_train_oof, self.__knn_test_oof = self.__knn_oof.get_oof()
        print("--- knn ready ! ---")

        self.__et_oof = OutOfFold(clf=self.__et_estimator, train=self.__train, train_label=self.__train_label,
                                  test=self.__test, cv=self.__cv, random_state=self.__random_state)
        self.__et_oof.set_skf()
        self.__et_train_oof, self.__et_test_oof = self.__et_oof.get_oof()
        print("--- et ready ! ---")

        self.__ad_oof = OutOfFold(clf=self.__ad_estimator, train=self.__train, train_label=self.__train_label,
                                  test=self.__test, cv=self.__cv, random_state=self.__random_state)
        self.__ad_oof.set_skf()
        self.__ad_train_oof, self.__ad_test_oof = self.__ad_oof.get_oof()
        print("--- ad ready ! ---")

        self.__gbdt_oof = OutOfFold(clf=self.__gbdt_estimator, train=self.__train, train_label=self.__train_label,
                                    test=self.__test, cv=self.__cv, random_state=self.__random_state)
        self.__gbdt_oof.set_skf()
        self.__gbdt_train_oof, self.__gbdt_test_oof = self.__gbdt_oof.get_oof()
        print("--- gbdt ready ! ---")

        self.__xgb_oof = OutOfFold(clf=self.__xgb_estimator, train=self.__train, train_label=self.__train_label,
                                   test=self.__test, cv=self.__cv, random_state=self.__random_state)
        self.__xgb_oof.set_skf()
        self.__xgb_train_oof, self.__xgb_test_oof = self.__xgb_oof.get_oof()
        print("--- xgb ready ! ---")

        self.__rf_oof = OutOfFold(clf=self.__rf_estimator, train=self.__train, train_label=self.__train_label,
                                  test=self.__test, cv=self.__cv, random_state=self.__random_state)
        self.__rf_oof.set_skf()
        self.__rf_train_oof, self.__rf_test_oof = self.__rf_oof.get_oof()
        print("--- rf ready ! ---")

    def aggr_feature(self):
        mda = (ModelDataAgg(self.__train, self.__test,
                            self.__lr_train_oof.shape[0], self.__lr_test_oof.shape[0],
                            lr_train_oof=self.__lr_train_oof, lr_test_oof=self.__lr_test_oof,
                            knn_train_oof=self.__knn_train_oof, knn_test_oof=self.__knn_test_oof,
                            et_train_oof=self.__et_train_oof, et_test_oof=self.__et_test_oof,
                            ad_train_oof=self.__ad_train_oof, ad_test_oof=self.__ad_test_oof,
                            gbdt_train_oof=self.__gbdt_train_oof, gbdt_test_oof=self.__gbdt_test_oof,
                            xgb_train_oof=self.__xgb_train_oof, xgb_test_oof=self.__xgb_test_oof,
                            rf_train_oof=self.__rf_train_oof, rf_test_oof=self.__rf_test_oof))
        mda.train_test_split()
        self.__train_all = mda.train_merge()
        self.__test_all = mda.test_merge()

        return self.__train_all, self.__test_all