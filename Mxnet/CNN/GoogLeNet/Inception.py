# coding:utf-8

from mxnet.gluon import nn
from mxnet import nd


class Inception(nn.Block):
    # n1_1, n2_1, n2_3, n3_1, n3_5, n4_1 均是 output_channels/num_filters
    # Inception output_channels/num_filters n1_1 + n2_3 + n3_5 + n4_1
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # path 1
        self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1,
                                   activation='relu')
        # path 2
        self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1,
                                   activation='relu')
        self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1,
                                   activation='relu')
        # path 3
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1,
                                   activation='relu')
        self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2,
                                   activation='relu')
        # path 4
        self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1,
                                      strides=1)
        self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1,
                                   activation='relu')

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))

        return nd.concat(p1, p2, p3, p4, dim=1)