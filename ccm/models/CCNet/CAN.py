import torch.nn as nn
from ccm.models.backbones.vgg_10_d8 import vgg_block
# from ccm.models.backbones.vgg_10_d8_org import vgg_block
from cccv.cnn.layers import BaseConv, BaseDeConv
import torch.nn.functional as F
from config.sys_config_back import sys_cfg
from .custom_BN import MixBatchNorm2d, SplitBatchNorm2d
import torch

if sys_cfg.bn_dtype == 'mix':
    norm_layer = MixBatchNorm2d
elif sys_cfg.bn_dtype == 'split':
    norm_layer = SplitBatchNorm2d
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
            else:
                layers += [conv2d, nn.ReLU()]
            if dropout:
                layers += [nn.Dropout2d(p=sys_cfg.dropout_rate)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNetBackbone(nn.Module):

    def __init__(self, load_weights=False, reduce_channel=1, bn=True):
        super(CSRNetBackbone, self).__init__()
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel, bn=bn)

        self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
        self.backend = make_layers(self.backend_feat,
                                   in_channels=512 // reduce_channel,
                                   batch_norm=bn,
                                   dilation=True,
                                   dropout=sys_cfg.enable_drop)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


class CANet(nn.Module):
    '''
    this is a baseline model for MATT method, it only use small labeled data to train model.
    '''

    def __init__(self, load_weights=False, reduce_channel=1, bn=True):
        super(CANet, self).__init__()
        self.primary_branch_feat = [128 // reduce_channel, 64 // reduce_channel]  # 主分支各卷积层的通道数
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel, bn=bn)
        self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel,
                             256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
        self.backend = make_layers(self.backend_feat,
                                   in_channels=1024 // reduce_channel,
                                   batch_norm=bn,
                                   dilation=True,
                                   dropout=sys_cfg.enable_drop)

        # 定义最后一层卷积网络，1*1,不使用激活函数，方便优化。
        self.output_layer = BaseConv(in_channels=self.primary_branch_feat[-1],
                                     out_channels=1,
                                     kernel_size=1,
                                     NL='relu')
        # define the canet decoder
        self.conv1_1 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')
        self.conv1_2 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')
        self.conv2_1 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')
        self.conv2_2 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')
        self.conv3_1 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')
        self.conv3_2 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')
        self.conv6_1 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')
        self.conv6_2 = BaseConv(in_channels=512, out_channels=512, kernel_size=1, NL='None')

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

        fv = self.frontend(x)
        # x += 10

        ave1 = F.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)

        s1 = F.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        C1 = s1 - fv

        w1 = self.conv1_2(C1)
        w1 = torch.sigmoid(w1)

        # S=2
        ave2 = F.adaptive_avg_pool2d(fv, (1, 1))
        ave2 = self.conv2_1(ave2)

        s2 = F.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        C2 = s2 - fv

        w2 = self.conv2_2(C2)
        w2 = torch.sigmoid(w2)

        # S=3
        ave3 = F.adaptive_avg_pool2d(fv, (1, 1))
        ave3 = self.conv3_1(ave3)

        s3 = F.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        C3 = s3 - fv

        w3 = self.conv3_2(C3)
        w3 = torch.sigmoid(w3)

        # S=6

        ave6 = F.adaptive_avg_pool2d(fv, (1, 1))
        ave6 = self.conv6_1(ave6)

        s6 = F.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        C6 = s6 - fv

        w6 = self.conv6_2(C6)
        w6 = torch.sigmoid(w6)
        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 1e-20)
        x = torch.cat([fv, fi],1)

        x = self.backend(x)
        x = self.output_layer(x)

        return x
