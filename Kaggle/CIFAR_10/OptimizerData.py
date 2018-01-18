# coding:utf-8

import os
from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np


class OptimizerData(object):

    def __init__(self, *, input_path, folder_list, batch_size):
        self.__input_path = input_path
        self.__folder_list = folder_list
        self.__batch_size = batch_size
        self.__train_ds, self.__valid_ds, self.__train_valid_ds, self.__test_ds = [None for _ in range(4)]
        self.__train_data, self.__valid_data, self.__train_valid_data, self.__test_data = [None for _ in range(4)]

    def load_data(self):
        def transform_train(data, label):
            im = data.astype("float32") / 255
            auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                                            rand_crop=False, rand_resize=False, rand_mirror=True,
                                            mean=np.array([0.4914, 0.4822, 0.4465]),
                                            std=np.array([0.2023, 0.1994, 0.2010]),
                                            brightness=0, contrast=0,
                                            saturation=0, hue=0,
                                            pca_noise=0, rand_gray=0, inter_method=2)
            for aug in auglist:
                im = aug(im)
            # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
            im = nd.transpose(im, (2, 0, 1))

            return im, nd.array([label]).asscalar().astype("float32")

        # 测试时，无需对图像做标准化以外的增强数据处理。
        def transform_test(data, label):
            im = data.astype("float32") / 255
            auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                                            mean=np.array([0.4914, 0.4822, 0.4465]),
                                            std=np.array([0.2023, 0.1994, 0.2010]))
            for aug in auglist:
                im = aug(im)
            im = nd.transpose(im, (2, 0, 1))

            return im, nd.array([label]).asscalar().astype("float32")

        # 读取原始图像文件。flag=1 说明输入图像有三个通道（彩色）。
        self.__train_ds = vision.ImageFolderDataset(
            os.path.join(self.__input_path, self.__folder_list[0]), flag=1, transform=transform_train)
        self.__valid_ds = vision.ImageFolderDataset(
            os.path.join(self.__input_path, self.__folder_list[1]), flag=1, transform=transform_test)
        self.__train_valid_ds = vision.ImageFolderDataset(
            os.path.join(self.__input_path, self.__folder_list[2]), flag=1, transform=transform_train)
        self.__test_ds = vision.ImageFolderDataset(
            os.path.join(self.__input_path, self.__folder_list[3]), flag=1, transform=transform_test)

        # print(len(self.__train_ds)) 样本个数

        self.__train_data = gluon.data.DataLoader(
            self.__train_ds, self.__batch_size, shuffle=True, last_batch="keep")
        self.__valid_data = gluon.data.DataLoader(
            self.__valid_ds, self.__batch_size, shuffle=True, last_batch="keep")
        self.__train_valid_data = gluon.data.DataLoader(
            self.__train_valid_ds, self.__batch_size, shuffle=True, last_batch="keep")
        self.__test_data = gluon.data.DataLoader(
            self.__test_ds, self.__batch_size, shuffle=False, last_batch="keep")

        return self.__train_data, self.__valid_data, self.__train_valid_data, self.__test_data


if __name__ == "__main__":
    od = OptimizerData(input_path="D:\\Code\\kaggle\\cifar10\\train_valid_test",
                      folder_list=["train", "valid", "train_valid", "test"],
                      batch_size=128)

    train_data, valid_data, train_valid_data, test_data = od.load_data()

    for batch_X, batch_y in train_data:
        print(batch_X.shape)
        print(batch_y.shape)
        break



