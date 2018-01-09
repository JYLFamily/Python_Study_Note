# coding:utf-8

from mxnet.gluon import nn

from Mxnet.CNN.ResNet.Residual import Residual


class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                Residual(128, same_shape=False),
                Residual(128)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                Residual(256, same_shape=False),
                Residual(256)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                Residual(512, same_shape=False),
                Residual(512)
            )
            # block 6
            b6 = nn.Sequential()
            b6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out