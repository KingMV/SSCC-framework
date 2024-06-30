import torch.nn as nn
from ccm.models.backbones.vgg_10_d8 import vgg_block
# from ccm.models.backbones.vgg_10_d8_org import vgg_block
from cccv.cnn.layers import BaseConv, BaseDeConv
import torch.nn.functional as F
import torch


class Crowd_SPN(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(Crowd_SPN, self).__init__()
        self.save_features = True
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        if reduce_channel == 1:
            self.backend_feat01 = [512, 512, 512]
            self.backend_feat02 = [256, 128, 64]
        else:
            self.backend_feat01 = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
            self.backend_feat02 = [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
            # [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
        self.sam = SAM(inchannel=512 // reduce_channel,
                       dilation_ratio_list=[1, 2, 4, 8],
                       # dilation_ratio_list=[2, 4, 8, 12],
                       branch_number=4,
                       planes_channel=256)
        self.backend = make_layers(self.backend_feat01, in_channels=512 // reduce_channel, batch_norm=True,
                                   dilation=True)

        # self.classifier = nn.Sequential(
        #     BaseConv(512 // reduce_channel, 128 // reduce_channel, kernel_size=3, bn=True),
        #     BaseConv(128 // reduce_channel, 64 // reduce_channel, kernel_size=3, bn=True),
        #     nn.AdaptiveAvgPool2d(1),
        #     BaseConv(64 // reduce_channel, 4, kernel_size=1, NL='None', bn=False)
        # )
        self.regressor = nn.Sequential(
            make_layers(self.backend_feat02, in_channels=512 // reduce_channel, batch_norm=True, dilation=True),
            BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None', bn=False)
        )

    def forward(self, x):
        self.features = []
        features = self.frontend(x)
        x = features[-1]
        x = self.sam(x)
        x = self.backend(x)
        y1 = self.regressor(x)
        # y2 = self.classifier(x)
        # y2 = x.view(y2.size(0), -1)
        return y1

    def enable_features(self, flag):
        self.save_features = flag

    def regist_hook(self):
        self.features = []

        def get(model, input, output):
            # function will be automatically called each time, since the hook is injected
            if self.save_features:
                self.features.append(output.detach().cpu())

        # for name, module in self.named_modules():
        #     print(name)
        #     print(module)

        for name, module in self._modules['frontend']._modules.items():
            if name in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']:
                self._modules['frontend']._modules[name].register_forward_hook(get)
                print(name)
                print(module)
        for name, module in self._modules['backend']._modules.items():
            if name in ['8']:
                self._modules['backend']._modules[name].register_forward_hook(get)
                print(name)
                print(module)


class SAM(nn.Module):
    def __init__(self, inchannel, dilation_ratio_list, branch_number=3, planes_channel=256):
        super(SAM, self).__init__()
        self.red_ratio = inchannel // planes_channel

        self.branch_1 = nn.Sequential(
            BaseConv(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                     dilation=1),
            BaseConv(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                     same_padding=True, bn=True, dilation=dilation_ratio_list[0])
        )
        self.branch_2 = nn.Sequential(
            BaseConv(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                     dilation=1),
            BaseConv(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                     same_padding=True, bn=True, dilation=dilation_ratio_list[1])
        )
        self.branch_3 = nn.Sequential(
            BaseConv(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                     dilation=1),
            BaseConv(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                     same_padding=True, bn=True, dilation=dilation_ratio_list[2])
        )
        self.branch_4 = nn.Sequential(
            BaseConv(inchannel, inchannel // self.red_ratio, kernel_size=1, stride=1, same_padding=True, bn=True,
                     dilation=1),
            BaseConv(inchannel // self.red_ratio, inchannel // self.red_ratio, kernel_size=3, stride=1,
                     same_padding=True, bn=True, dilation=dilation_ratio_list[3])
        )
        self.fuse = BaseConv(inchannel // self.red_ratio * branch_number, inchannel, kernel_size=3, stride=1,
                             same_padding=True, bn=True)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        # x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fuse(x)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
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
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
