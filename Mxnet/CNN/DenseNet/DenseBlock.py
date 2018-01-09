# coding:utf-8

from mxnet import nd
from mxnet.gluon import nn


def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation("relu"),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out


class DenseBlock(nn.Block):
    # layers 这个 DenseBlock 中包含 layer 个 conv_block
    # 每个 layers 的 output_channels 是 growth_rate
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x