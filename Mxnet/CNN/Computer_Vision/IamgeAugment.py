# coding:utf-8

import matplotlib.pyplot as plt
from mxnet import nd
from mxnet import image


class IamgeAugment(object):
    def __init__(self, *, input_path, n):
        self.__img = image.imdecode(open(input_path, "rb").read())
        self.__n = n
        self.__X = None
        self.__Y = None

    def apply(self, *, aug):
        def show_images(imgs, nrows, ncols, figsize=None):
            """plot a list of images"""
            if not figsize:
                figsize = (ncols, nrows)
            _, figs = plt.subplots(nrows, ncols, figsize=figsize)
            for i in range(nrows):
                for j in range(ncols):
                    figs[i][j].imshow(imgs[i * ncols + j].asnumpy())
                    figs[i][j].axes.get_xaxis().set_visible(False)
                    figs[i][j].axes.get_yaxis().set_visible(False)
            plt.show()

        self.__X = [aug(self.__img.astype("float32")) for _ in range(self.__n * self.__n)]
        self.__Y = nd.stack(* self.__X).clip(0, 255) / 255
        show_images(self.__Y, self.__n, self.__n, figsize=(8, 8))


if __name__ == "__main__":
    ia = IamgeAugment(input_path="cat.jpg", n=3)
    # 变形
    # 50% 几率水平翻转图片
    ia.apply(aug=image.HorizontalFlipAug(.5))
    # 剪裁
    # 随机从原图片剪裁出 200×200 的一部分 , 可以图片先变大在剪小
    ia.apply(aug=image.RandomCropAug([200, 200]))
    # 随机从图片中剪裁一个区域要求是剪裁面积占原图片 >= 0.1 且剪裁的区域长/宽在 0.5~2 之间后将剪裁后的图片放缩到 200×200
    ia.apply(aug=image.RandomSizedCropAug((200, 200), .1, (.5, 2)))

    # 色调
    # 随机将亮度增加或者减小在 0-50% 间的一个量
    ia.apply(aug=image.BrightnessJitterAug(.5))
    # 随机色调变化
    ia.apply(aug=image.HueJitterAug(.5))