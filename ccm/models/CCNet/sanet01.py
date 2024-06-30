from cccv.cnn.layers import BaseConv, BaseDeConv
import torch
import torch.nn as nn


class SAM_Head(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAM_Head, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BaseConv(in_channels, branch_out, kernel_size=1, bn=use_bn)
        self.branch3x3 = BaseConv(in_channels, branch_out, kernel_size=3, bn=use_bn)
        self.branch5x5 = BaseConv(in_channels, branch_out, kernel_size=5, bn=use_bn)
        self.branch7x7 = BaseConv(in_channels, branch_out, kernel_size=7, bn=use_bn)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SAModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BaseConv(in_channels, branch_out, kernel_size=1, bn=use_bn)
        self.branch3x3 = nn.Sequential(
            BaseConv(in_channels, 2 * branch_out, kernel_size=1, bn=use_bn),
            BaseConv(2 * branch_out, branch_out, kernel_size=3, bn=use_bn),
        )
        self.branch5x5 = nn.Sequential(
            BaseConv(in_channels, 2 * branch_out, kernel_size=1, bn=use_bn),
            BaseConv(2 * branch_out, branch_out, kernel_size=5, bn=use_bn),
        )
        self.branch7x7 = nn.Sequential(
            BaseConv(in_channels, 2 * branch_out, kernel_size=1, bn=use_bn),
            BaseConv(2 * branch_out, branch_out, kernel_size=7, bn=use_bn),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SANet(nn.Module):
    def __init__(self, use_bn=True):
        super(SANet, self).__init__()

        in_channels = 3
        self.encoder = nn.Sequential(
            SAM_Head(in_channels, 64, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(64, 128, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(128, 128, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(128, 64, use_bn),
        )
        self.decoder = nn.Sequential(
            BaseConv(64, 64, bn=use_bn, kernel_size=9),
            BaseDeConv(64, 64, 2, stride=2, bn=use_bn),
            BaseConv(64, 32, bn=use_bn, kernel_size=7),
            BaseDeConv(32, 32, 2, stride=2, bn=use_bn),
            BaseConv(32, 16, bn=use_bn, kernel_size=5),
            BaseDeConv(16, 16, 2, stride=2, bn=use_bn),
            BaseConv(16, 16, bn=use_bn, kernel_size=3)
        )

        self.output_layer = BaseConv(16, 1, kernel_size=3, NL='None', bn=False)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        out = self.output_layer(out)
        return out
