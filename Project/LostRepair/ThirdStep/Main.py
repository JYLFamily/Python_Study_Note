# coding:utf-8

from Project.LostRepair.ThirdStep.RawDataSplit import RawDataSplit
from Project.LostRepair.ThirdStep.StackingAssistant import StackingAssistant
from Project.LostRepair.ThirdStep.XgbModel import XgbModel

if __name__ == "__main__":
    rds = RawDataSplit(input_path="D:\\Project\\LostRepair\\yunyingshang\\all.csv")
    rds.set_train_test()
    train, train_label, test, test_label = rds.get_train_test()
    print("------ RawDataSplit complete ! ------")

    sa = StackingAssistant(train=train, train_label=train_label, test=test, cv=5)
    sa.calc_feature()
    train, test = sa.aggr_feature()
    print("------ StackingAssistant complete ! ------")

    xgb = XgbModel(train=train, train_label=train_label, test=test, test_label=test_label)
    xgb.train()
    xgb.predict()
    xgb.evaluate()
    xgb.evaluate_output()
    xgb.xgbmodel_output()
    print("------ Model complete ! ------")