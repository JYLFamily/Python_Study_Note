# coding:utf-8

import os
import shutil
import zipfile


class TrainValidTest(object):
    def __init__(self,
                 *, zip_file_list=None, data_dir=None, label_file=None,
                 train_dir=None, test_dir=None, input_dir=None, valid_ratio=None):
        """
        :param zip_file_list: 待解压文件列表
        :param data_dir: 数据路径（压缩包文件所在路径）
        :param label_file: label file 文件名
        :param train_dir: train 文件夹名
        :param test_dir: test 文件夹名
        :param input_dir: 整理后文件所在文件夹名
        :param valid_ratio: 验证集数据所占训练集比例
        """
        self.__zip_file_list = zip_file_list
        self.__data_dir = data_dir
        self.__label_file = label_file
        self.__train_dir = train_dir
        self.__test_dir = test_dir
        self.__input_dir = input_dir
        self.__valid_ratio = valid_ratio

    def extract_from_zip(self):
        for fin in self.__zip_file_list:
            with zipfile.ZipFile(os.path.join(self.__data_dir, fin), "r") as zin:
                zin.extractall(self.__data_dir)

    def extract_from_7z(self):
        pass

    def reorg_cifar10_data(self):
        # 读取训练数据标签。
        with open(os.path.join(self.__data_dir, self.__label_file), "r") as f:
            # 跳过文件头行（header）。
            lines = f.readlines()[1:]
            tokens = [l.rstrip().split(',') for l in lines]
            # 字典 key 样本序号 value 样本类别
            idx_label = dict(((int(idx), label) for idx, label in tokens))
        # label_file 中共有多少个类别 , set
        labels = set(idx_label.values())

        # 训练集中样本个数
        num_train = len(os.listdir(os.path.join(self.__data_dir, self.__train_dir)))
        # 训练集去除验证集合后样本个数
        num_train_tuning = int(num_train * (1 - self.__valid_ratio))
        assert 0 < num_train_tuning < num_train
        # 训练集去除验证集合后样本平均到每个 label 样本个数
        num_train_tuning_per_label = num_train_tuning // len(labels)
        # key label value count 用于统计 train 文件夹中某个 label 样本个数是否能达到 num_train_tuning_per_label 个
        # 如果达到放入 train 文件夹 否则放入 valid 文件夹
        label_count = dict()

        def mkdir_if_not_exist(path):
            if not os.path.exists(os.path.join(*path)):
                os.makedirs(os.path.join(*path))

        # 整理训练和验证集。
        for train_file in os.listdir(os.path.join(self.__data_dir, self.__train_dir)):
            # 样本序号
            idx = int(train_file.split(".")[0])
            # 样本 label
            label = idx_label[idx]
            mkdir_if_not_exist([self.__data_dir, self.__input_dir, "train_valid", label])
            shutil.copy(os.path.join(self.__data_dir, self.__train_dir, train_file),
                        os.path.join(self.__data_dir, self.__input_dir, "train_valid", label))
            if label not in label_count or label_count[label] < num_train_tuning_per_label:
                mkdir_if_not_exist([self.__data_dir, self.__input_dir, "train", label])
                shutil.copy(os.path.join(self.__data_dir, self.__train_dir, train_file),
                            os.path.join(self.__data_dir, self.__input_dir, "train", label))
                label_count[label] = label_count.get(label, 0) + 1
            else:
                mkdir_if_not_exist([self.__data_dir, self.__input_dir, "valid", label])
                shutil.copy(os.path.join(self.__data_dir, self.__train_dir, train_file),
                            os.path.join(self.__data_dir, self.__input_dir, "valid", label))

        # 整理测试集。
        mkdir_if_not_exist([self.__data_dir, self.__input_dir, "test", "unknown"])
        for test_file in os.listdir(os.path.join(self.__data_dir, self.__test_dir)):
            shutil.copy(os.path.join(self.__data_dir, self.__test_dir, test_file),
                        os.path.join(self.__data_dir, self.__input_dir, "test", "unknown"))


if __name__ == "__main__":
    tvt = TrainValidTest(
        zip_file_list=["trainLabels.zip"],
        data_dir="D:\\Code\\kaggle\\cifar10",
        label_file="trainLabels.csv",
        train_dir="train",
        test_dir="test",
        input_dir="train_valid_test",
        valid_ratio=0.1
    )
    # tvt.extract_from_zip()
    tvt.reorg_cifar10_data()