# coding:utf-8

from mxnet.gluon import nn
from mxnet import nd


class Residual(nn.Block):
    # channels
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(
            # same_shape
            # conv kernel_size=3 padding=1 strides=1
            # conv kernel_size=3 padding=1 strides=1
            # no same_shape
            # conv kernel_size=3 padding=1 strides=2
            # conv kernel_size=3 padding=1 strides=1
            channels, kernel_size=3, padding=1, strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(
                channels, kernel_size=1, strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)

        return nd.relu(out + x)


if __name__ == "__main__":
    blk = Residual(channels=3, same_shape=False)
    blk.initialize()
    x = nd.random.uniform(shape=(4, 3, 6, 6))
    blk(x)