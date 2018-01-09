# coding:utf-8

from mxnet.gluon import nn
from Mxnet.CNN.DenseNet.DenseBlock import DenseBlock
from Mxnet.CNN.DenseNet.TransitionBlock import transition_block


init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10


def dense_net():
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            # channels = channels + layers * growth_rate
            # 上面的 DenseBlock output_channels 是 channels = channels + layers * growth_rate
            # 后续使用 transition_block 将它的 channels 变为一半
            channels += layers * growth_rate
            # 最后一个 Dense Block 后面不加 transition_block
            if i != len(block_layers)-1:
                net.add(transition_block(channels//2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net