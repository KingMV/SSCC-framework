# -*-coding:utf-8-*-
import torch.nn as nn
from ccm.models.backbones.vgg_10_d8 import vgg_block
from ccm.models.backbones.VGG import vgg10_backbone
from cccv.cnn.layers import BaseConv
import torch.nn.functional as F
from config.sys_config_back import sys_cfg
from functools import partial
from .custom_BN import MixBatchNorm2d, SplitBatchNorm2d, DSBN
import torch


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


if sys_cfg.bn_dtype == 'mix':
    norm_layer = MixBatchNorm2d
elif sys_cfg.bn_dtype == 'split':
    norm_layer = SplitBatchNorm2d
elif sys_cfg.bn_dtype == 'dsbn':
    norm_layer = DSBN
else:
    norm_layer = nn.BatchNorm2d


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False, dropout=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, norm_layer(v), nn.ReLU()]
                # layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            if dropout:
                layers += [nn.Dropout2d(p=sys_cfg.dropout_rate)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNetBackbone(nn.Module):

    def __init__(self, load_weights=False, reduce_channel=1, bn=True):
        super(CSRNetBackbone, self).__init__()
        # self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel, bn=bn)
        self.frontend = vgg10_backbone(enable_bn=True, load_imagenet_weight=True)
        self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
        self.backend = make_layers(self.backend_feat,
                                   in_channels=512 // reduce_channel,
                                   batch_norm=bn,
                                   dilation=True,
                                   dropout=False)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


class Baseline1(nn.Module):
    '''
    this is a baseline model for MATT method, it only use small labeled data to train model.
    '''

    def __init__(self, load_weights=False, reduce_channel=1):
        super(Baseline1, self).__init__()
        self.primary_branch_feat = [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]  # 主分支各卷积层的通道数
        self.csrnet_backbone = CSRNetBackbone(load_weights, reduce_channel)  # CSRNet的主干网络部分，用于提取特征。
        self.primary_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=True,
                                          dropout=False)
        # 定义最后一层卷积网络，1*1,不使用激活函数，方便优化。
        self.output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                     out_channels=1,
                                     kernel_size=1,
                                     NL='relu')
        if sys_cfg.use_uncertainty:
            self.var_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=False,
                                          dropout=False)
            self.var_output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                             out_channels=1,
                                             kernel_size=1,
                                             NL='None'
                                             )
        else:
            self.var_output_layer = None

    def forward(self, x):

        x = self.csrnet_backbone(x)
        # x += 10

        if sys_cfg.output_feature:
            temp = []
            temp.append(x)
        # y1 = self.primary_branch(x)
        x = self.primary_branch(x)
        # x += torch.clamp(torch.randn_like(x) * 0.1, -0.5, 0.5)
        # x -= 0.5
        # print('max is {0},min is {1}'.format(x.max(), x.min()))
        # d_max = x.max(keepdim=True, dim=3)[0].max(keepdim=True, dim=2)[0].max(keepdim=True, dim=1)[0]
        # d_min = x.min(keepdim=True, dim=3)[0].min(keepdim=True, dim=2)[0].min(keepdim=True, dim=1)[0]
        # x = (x - d_min) / (d_max - d_min)
        # x = x - torch.mean(x, keepdim=True, dim=1)
        # x = x / torch.var(x, keepdim=True, dim=1)
        # x += 0.2
        y1 = self.output_layer(x)
        if self.var_output_layer is not None:
            # y2 = self.var_branch(x)
            y2 = self.var_output_layer(x)
            return y1, y2
        else:
            if sys_cfg.output_feature:
                return y1, temp
            return y1


class Baseline2(nn.Module):
    '''
    this is a baseline model for MATT method, it only use small labeled data to train model.
    '''

    def __init__(self, load_weights=False, reduce_channel=1):
        super(Baseline2, self).__init__()
        self.primary_branch_feat = [128 // reduce_channel, 64 // reduce_channel]  # 主分支各卷积层的通道数
        self.csrnet_backbone = CSRNetBackbone(load_weights, reduce_channel, bn=False)  # CSRNet的主干网络部分，用于提取特征。
        self.primary_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=False,
                                          dilation=False,
                                          dropout=False)
        # 定义最后一层卷积网络，1*1,不使用激活函数，方便优化。
        self.output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                     out_channels=1,
                                     kernel_size=1,
                                     NL='relu')
        if sys_cfg.use_uncertainty:
            self.var_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=False,
                                          dilation=False,
                                          dropout=False)
            self.var_output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                             out_channels=1,
                                             kernel_size=1,
                                             NL='None'
                                             )
        else:
            self.var_output_layer = None

    def forward(self, x):

        x = self.csrnet_backbone(x)
        if sys_cfg.output_feature:
            temp = []
            temp.append(x)
        # y1 = self.primary_branch(x)
        x = self.primary_branch(x)
        y1 = self.output_layer(x)
        if self.var_output_layer is not None:
            # y2 = self.var_branch(x)
            y2 = self.var_output_layer(x)
            return y1, y2
        else:
            if sys_cfg.output_feature:
                return y1, temp
            return y1


class UCB(nn.Module):
    '''
    this is a baseline model for MATT method, it only use small labeled data to train model.
    '''

    def __init__(self, load_weights=False, reduce_channel=1):
        super(UCB, self).__init__()
        self.primary_branch_feat = [128 // reduce_channel, 64 // reduce_channel]  # 主分支各卷积层的通道数
        self.csrnet_backbone = CSRNetBackbone(load_weights, reduce_channel)  # CSRNet的主干网络部分，用于提取特征。
        self.primary_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=False,
                                          dropout=False)
        # 定义最后一层卷积网络，1*1,不使用激活函数，方便优化。
        self.output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                     out_channels=1,
                                     kernel_size=1,
                                     NL='relu')
        self.var_branch = make_layers(self.primary_branch_feat,
                                      in_channels=512 // reduce_channel,
                                      batch_norm=True,
                                      dilation=False,
                                      dropout=False)
        self.var_output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                         out_channels=1,
                                         kernel_size=1,
                                         NL='None'
                                         )

    def forward(self, x):
        x = self.csrnet_backbone(x)
        y1 = self.primary_branch(x)
        y1 = self.output_layer(y1)
        if self.var_output_layer is not None:
            y2 = self.var_branch(x)
            y2 = self.var_output_layer(y2)
            return y1, y2
        else:
            return y1


class MDN(nn.Module):
    '''
    this is a baseline model for MATT method, it only use small labeled data to train model.
    '''

    def __init__(self, load_weights=False, reduce_channel=1):
        super(MDN, self).__init__()
        K = 10
        self.primary_branch_feat = [128 // reduce_channel, 64 // reduce_channel]  # 主分支各卷积层的通道数
        self.csrnet_backbone = CSRNetBackbone(load_weights, reduce_channel)  # CSRNet的主干网络部分，用于提取特征。
        self.primary_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=False,
                                          dropout=False)
        # 定义最后一层卷积网络，1*1,不使用激活函数，方便优化。
        self.output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                     out_channels=1,
                                     kernel_size=1,
                                     NL='None')
        if sys_cfg.use_uncertainty:
            self.var_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=False,
                                          dropout=False)
            self.sigma = BaseConv(in_channels=self.primary_branch_feat[-1],
                                  out_channels=K,  # gaussian number
                                  kernel_snize=1,
                                  NL='None')
            self.mu = BaseConv(in_channels=self.primary_branch_feat[-1],
                               out_channels=K,  # gaussian number
                               kernel_size=1,
                               NL='None')
            self.pi = BaseConv(in_channels=self.primary_branch_feat[-1],
                               out_channels=K,  # gaussian number
                               kernel_size=1,
                               NL='None')
        else:
            self.var_output_layer = None

    def forward(self, x):
        x = self.csrnet_backbone(x)
        y1 = self.primary_branch(x)
        y1 = self.output_layer(y1)
        if sys_cfg.use_uncertainty:
            y2 = self.var_branch(x)
            sigma = self.sigma(y2)
            mu = self.mu(y2)
            pi = self.pi(y2)
            return y1, [mu, sigma, pi]
        else:
            return y1


class CSeg(nn.Module):
    '''
    this is a baseline model for MATT method, it only use small labeled data to train model.
    '''

    def __init__(self, load_weights=False, reduce_channel=1):
        super(CSeg, self).__init__()
        self.primary_branch_feat = [128 // reduce_channel, 64 // reduce_channel]  # 主分支各卷积层的通道数
        self.csrnet_backbone = CSRNetBackbone(load_weights, reduce_channel)  # CSRNet的主干网络部分，用于提取特征。
        self.primary_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=False,
                                          dropout=False)
        # 定义最后一层卷积网络，1*1,不使用激活函数，方便优化。
        self.output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                     out_channels=1,
                                     kernel_size=1,
                                     NL='None')
        if sys_cfg.use_seg_branch:
            self.seg_branch_feat = [128 // reduce_channel, 64 // reduce_channel]
            self.seg_branch = make_layers(self.seg_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=False,
                                          dropout=False)
            self.seg_output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                             out_channels=1,
                                             kernel_size=1,
                                             NL='sigmoid'
                                             )
        else:
            self.seg_output_layer = None

    def forward(self, x):
        x = self.csrnet_backbone(x)
        y1 = self.primary_branch(x)
        y1 = self.output_layer(y1)
        if self.seg_output_layer is not None:
            y2 = self.seg_branch(x)
            y2 = self.seg_output_layer(y2)
            return y1, y2
        else:
            return y1

    def forward(self, x):
        x = self.csrnet_backbone(x)
        y1 = self.primary_branch(x)
        y1 = self.output_layer(y1)
        if self.seg_output_layer is not None:
            y2 = self.seg_branch(x)
            y2 = self.seg_output_layer(y2)
            return y1, y2
        else:
            return y1


class SWDual_Network(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(SWDual_Network, self).__init__()
        self.primary_branch_feat = [128 // reduce_channel, 64 // reduce_channel]  # 主分支各卷积层的通道数
        self.csrnet_backbone = CSRNetBackbone(load_weights, reduce_channel)  # CSRNet的主干网络部分，用于提取特征。
        self.primary_branch = make_layers(self.primary_branch_feat,
                                          in_channels=512 // reduce_channel,
                                          batch_norm=True,
                                          dilation=False,
                                          dropout=sys_cfg.enable_drop)
        self.second_branch = make_layers(self.primary_branch_feat,
                                         in_channels=512 // reduce_channel,
                                         batch_norm=True,
                                         dilation=False,
                                         dropout=sys_cfg.enable_drop)
        # 定义最后一层卷积网络，1*1,不使用激活函数，方便优化。
        self.layer_1 = BaseConv(in_channels=self.primary_branch_feat[-1],
                                out_channels=1,
                                kernel_size=1,
                                NL='None')

        self.layer_2 = BaseConv(in_channels=self.primary_branch_feat[-1],
                                out_channels=1,
                                kernel_size=1,
                                NL='None')

    def forward(self, x):
        x = self.csrnet_backbone(x)
        y1 = self.primary_branch(x)
        y1 = self.layer_1(y1)

        y2 = self.second_branch(x)
        y2 = self.layer_2(y2)

        return y1, y2
