import torch.nn as nn
from ccm.models.backbones.vgg_10_d8 import vgg_block
# from ccm.models.backbones.vgg_10_d8_org import vgg_block
from cccv.cnn.layers import BaseConv

channel_nums = [[32, 64, 128, 256],  # half
                [21, 43, 85, 171],  # third
                [16, 32, 64, 128],  # quarter
                [13, 26, 51, 102],  # fifth
                ]


class CSRNet(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1, transform=True):
        super(CSRNet, self).__init__()
        # self.frontend = vgg_block(pretrained=load_weights)

        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.frontend = make_layers(self.frontend_feat)
        self.transform = transform
        if reduce_channel == 1:
            self.backend_feat = [512, 512, 512, 256, 128, 64]
        else:
            self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel,
                                 256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
            # self.transform = True
        if reduce_channel == 1 and self.transform:
            raise ValueError('it dose not need transform')
        # self.transform = False
        self.backend_1 = BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, dilation=2, bn=True)
        self.backend_2 = nn.Sequential(
            BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(512 // reduce_channel, 256 // reduce_channel, 3, 1, dilation=2, bn=True)
        )
        self.backend_3 = nn.Sequential(
            BaseConv(256 // reduce_channel, 128 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(128 // reduce_channel, 64 // reduce_channel, 3, 1, dilation=2, bn=True)
        )
        if self.transform:
            channel = channel_nums[4 - 2]
            self.transform_1_2 = feature_transform(channel[0], 64)
            self.transform_2_2 = feature_transform(channel[1], 128)
            self.transform_3_3 = feature_transform(channel[2], 256)
            self.transform_4_3 = feature_transform(channel[3], 512)
            self.transform_5_1 = feature_transform(channel[3], 512)
            self.transform_5_3 = feature_transform(channel[2], 512)

        # self.backend = make_layers(self.backend_feat, in_channels=512 // reduce_channel, batch_norm=True, dilation=True)
        self.output_layer = BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None')

    def forward(self, x):
        self.features = []
        features = self.frontend(x)
        if self.transform:
            self.features.append(self.transform_1_2(features[0]))
            self.features.append(self.transform_2_2(features[1]))
            self.features.append(self.transform_3_3(features[2]))
            self.features.append(self.transform_4_3(features[3]))
        x = features[-1]
        x = self.backend_1(x)
        if self.transform:
            self.features.append(self.transform_5_1(x))
        x = self.backend_2(x)
        if self.transform:
            self.features.append(self.transform_5_4(x))
        x = self.backend_3(x)
        x = self.output_layer(x)
        return x

    # def regist_hook(self):
    #     self.features = []
    #
    #     def get(model, input, output):
    #         # function will be automatically called each time, since the hook is injected
    #         self.features.append(output.detach())
    #
    #     for name, module in self._modules['frontend']._modules.items():
    #         if name in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']:
    #             self._modules['frontend']._modules[name].register_forward_hook(get)
    #             print(name)
    #             print(module)
    #     for name, module in self._modules['backend']._modules.items():
    #         if name in ['2', '11']:
    #             self._modules['backend']._modules[name].register_forward_hook(get)
    #             print(name)
    #             print(module)


class SSL_Model_S(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1, transform=True):
        super(SSL_Model_S, self).__init__()
        # self.frontend = vgg_block(pretrained=load_weights)

        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.frontend = make_layers(self.frontend_feat)
        self.transform = transform
        if reduce_channel == 1:
            self.backend_feat01 = [512, 512, 512]
            self.backend_feat02 = [256, 128, 64]
        else:
            self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel,
                                 256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]

            self.backend_feat01 = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
            self.backend_feat02 = [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]

            # self.transform = True
        if reduce_channel == 1 and self.transform:
            raise ValueError('it dose not need transform')
        # self.transform = False
        self.backend = nn.Sequential(
            BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, dilation=2, bn=True)
            # BaseConv(512 // reduce_channel, 256 // reduce_channel, 3, 1, dilation=2, bn=True)
        )

        self.classifier = nn.Sequential(
            BaseConv(512 // reduce_channel, 128 // reduce_channel, kernel_size=3, bn=True),
            BaseConv(128 // reduce_channel, 64 // reduce_channel, kernel_size=3, bn=True),
            nn.AdaptiveAvgPool2d(1),
            BaseConv(64 // reduce_channel, 4, kernel_size=1, NL='None', bn=False)
        )
        # self.regressor = nn.Sequential(
        #     make_layers(self.backend_feat02, in_channels=512 // reduce_channel, batch_norm=True, dilation=True),
        #     BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None', bn=False)
        # )
        self.regressor = nn.Sequential(
            BaseConv(512 // reduce_channel, 256 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(256 // reduce_channel, 128 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(128 // reduce_channel, 64 // reduce_channel, 3, 1, dilation=2, bn=True),
            BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None', bn=False)
        )
        if self.transform:
            channel = channel_nums[4 - 2]
            self.transform_1_2 = feature_transform(channel[0], 64)
            self.transform_2_2 = feature_transform(channel[1], 128)
            self.transform_3_3 = feature_transform(channel[2], 256)
            self.transform_4_3 = feature_transform(channel[3], 512)
            self.transform_5_3 = feature_transform(channel[3], 512)
            # self.transform_5_4 = feature_transform(channel[2], 512)

        # self.backend = make_layers(self.backend_feat, in_channels=512 // reduce_channel, batch_norm=True, dilation=True)
        self.output_layer = BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None')

    def forward(self, x):
        self.features = []
        features = self.frontend(x)
        if self.transform:
            self.features.append(self.transform_1_2(features[0]))
            self.features.append(self.transform_2_2(features[1]))
            self.features.append(self.transform_3_3(features[2]))
            self.features.append(self.transform_4_3(features[3]))
        x = features[-1]
        x = self.backend(x)
        if self.transform:
            self.features.append(self.transform_5_3(x))
        y1 = self.regressor(x)
        y2 = self.classifier(x)
        y2 = x.view(y2.size(0), -1)
        return y1, y2

    # def regist_hook(self):
    #     self.features = []
    #
    #     def get(model, input, output):
    #         # function will be automatically called each time, since the hook is injected
    #         self.features.append(output.detach())
    #
    #     for name, module in self._modules['frontend']._modules.items():
    #         if name in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']:
    #             self._modules['frontend']._modules[name].register_forward_hook(get)
    #             print(name)
    #             print(module)
    #     for name, module in self._modules['backend']._modules.items():
    #         if name in ['2', '11']:
    #             self._modules['backend']._modules[name].register_forward_hook(get)
    #             print(name)
    #             print(module)


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


def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


if __name__ == '__main__':
    import torchsummary

    m = CSRNet(load_weights=False)
    torchsummary.summary(m, (3, 224, 224))
