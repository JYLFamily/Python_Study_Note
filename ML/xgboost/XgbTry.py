# coding:utf-8

import xgboost as xgb
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


class XgbTry(object):
    """"第一课 Xgboost 小试牛刀"""
    def __init__(self, input_path, param, num_round):
        """初始化参数

        Parameters
        ----------
        :param input_path: string
            训练集、测试集所在路径
        :param param: dict
            超参数
        :param num_round: int
            boosting 迭代次数 , 也就是生成多少棵树
        """
        self.__train = xgb.DMatrix(input_path + 'agaricus.txt.train')
        self.__train_label = self.__train.get_label()
        self.__test = xgb.DMatrix(input_path + 'agaricus.txt.test')
        self.__test_label = self.__test.get_label()
        self.__param = param
        self.__num_round = num_round
        self.__bst = None
        self.__train_preds = None
        self.__train_predictions = None
        self.__test_preds = None
        self.__test_predictions = None

    def eda_train_test(self):
        print("Train nrow : " + str(self.__train.num_row()))
        print("Train ncol : " + str(self.__train.num_col()))
        print("Test nrow : " + str(self.__test.num_row()))
        print("Test ncol : " + str(self.__test.num_col()))

    def trian_model(self):
        self.__bst = xgb.train(self.__param, self.__train, self.__num_round)

    def evaluate_train(self):
        # 因为 'objective':'binary:logistic' 所以输出的是 "是毒蘑菇的概率"
        self.__train_preds = self.__bst.predict(self.__train)
        # 四舍五入得到是否是毒蘑菇
        self.__train_predictions = [round(value) for value in self.__train_preds]
        print("Train Accuary : %.2f%%" % (accuracy_score(self.__train_label, self.__train_predictions) * 100))

    def evaluate_test(self):
        self.__test_preds = self.__bst.predict(self.__test)
        self.__test_predictions = [round(value) for value in self.__test_preds]
        print("Test Accuary : %.2f%%" % (accuracy_score(self.__test_label, self.__test_predictions) * 100))

    def tree_viz(self):
        # num_trees 显示第几课树 , 从 0 开始这里显示第一棵树
        # rankdir 'UT' 垂直显示 'LR' 水平显示
        xgb.plot_tree(self.__bst, num_trees=0, rankdir= 'UT')
        pyplot.show()


if __name__ == "__main__":
    param = {'max_depth':2,
             'eta':1,
             'silent':0,
             'objective':'binary:logistic'}
    num_round = 2

    xgb_try = XgbTry(input_path='C:\\Users\\YL\\Desktop\\', param=param, num_round=num_round)
    xgb_try.eda_train_test()
    xgb_try.trian_model()
    xgb_try.evaluate_train()
    xgb_try.evaluate_test()
    xgb_try.tree_viz()