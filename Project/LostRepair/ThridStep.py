# coding:utf-8

from Project.LostRepair.RawDataSplit import RawDataSplit
from Project.LostRepair.StackingAssistant import StackingAssistant
from Project.LostRepair.XgbModel import XgbModel


if __name__ == "__main__":
    rds = RawDataSplit(input_path="C:\\Users\\Dell\\Desktop\\zytsl_robot.csv")
    rds.set_train_test()
    train, train_label, test, test_label = rds.get_train_test()
    print("------ RawDataSplit complete ! ------")

    sa = StackingAssistant(train=train, train_label=train_label, test=test)
    sa.calc_feature()
    train, test = sa.aggr_feature()
    print("------ StackingAssistant complete ! ------")
    print(train[0:10])
    print("---------")
    print(test[0:10])