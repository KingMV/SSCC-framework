import torch.nn as nn
from ccm.models.backbones.vgg_10_d8 import vgg_block
# from ccm.models.backbones.vgg_10_d8_org import vgg_block
from cccv.cnn.layers import BaseConv, BaseDeConv
import torch.nn.functional as F


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


class FE(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(FE, self).__init__()
        # self.frontend = vgg_block(pretrained=load_weights)
        self.save_features = True
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        if reduce_channel == 1:
            self.backend_feat = [512, 512, 512]
        else:
            self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
        self.backend = make_layers(self.backend_feat, in_channels=512 // reduce_channel, batch_norm=True, dilation=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rank_fc = nn.Sequential(
            BaseConv(512 // reduce_channel, 256 // reduce_channel, kernel_size=1, bn=True),
            BaseConv(256 // reduce_channel, 2, kernel_size=1, NL='None', bn=False)
        )

    def forward(self, x):
        self.features = []
        features = self.frontend(x)
        x = features[-1]
        x = self.backend(x)
        x = self.avg_pool(x)
        x = self.rank_fc(x)
        return x

    def enable_features(self, flag):
        self.save_features = flag

    # def get(self, model, input, output):
    #     # function will be automatically called each time, since the hook is injected
    #     self.features.append(output.detach())

    def regist_hook(self):
        self.features = []

        def get(model, input, output):
            # function will be automatically called each time, since the hook is injected
            if self.save_features:
                self.features.append(output.detach())

        for name, module in self._modules['frontend']._modules.items():
            if name in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']:
                self._modules['frontend']._modules[name].register_forward_hook(get)
                print(name)
                print(module)
        for name, module in self._modules['backend']._modules.items():
            if name in ['11']:
                self._modules['backend']._modules[name].register_forward_hook(get)
                print(name)
                print(module)


class RankNet(nn.Module):
    def __init__(self, load_weights=False):
        super(RankNet, self).__init__()
        self.net = FE(load_weights=load_weights)

    def forward(self, x1):
        y1 = self.net(x1)
        return y1
