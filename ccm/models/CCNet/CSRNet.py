import torch.nn as nn
# from ccm.models.backbones.vgg_10_d8 import vgg_block
from ccm.models.backbones.VGG import vgg16_enc_backbone
from cccv.cnn.layers import BaseConv, BaseDeConv
import torch.nn.functional as F


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False, norm_layer=None, relu_layer=None):
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
                layers += [conv2d, norm_layer(v), relu_layer]
            else:
                layers += [conv2d, relu_layer]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, args=None):
        super(CSRNet, self).__init__()

        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=512, batch_norm=True, dilation=True,
                                   norm_layer=nn.BatchNorm2d, relu_layer=nn.ReLU(inplace=True))
        self.output_layer = BaseConv(64, 1, kernel_size=1, NL='None')
        self.frontend = vgg16_enc_backbone(enable_bn=True, load_imagenet_weight=True)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    # def regist_hook(self):
    #     self.features = []
    #
    #     def get(model, input, output):
    #         # function will be automatically called each time, since the hook is injected
    #         if self.save_features:
    #             self.features.append(output.detach())
    #
    #     for name, module in self._modules['frontend']._modules.items():
    #         if name in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']:
    #             self._modules['frontend']._modules[name].register_forward_hook(get)
    #             print(name)
    #             print(module)
    #     for name, module in self._modules['backend']._modules.items():
    #         if name in ['11']:
    #             self._modules['backend']._modules[name].register_forward_hook(get)
    #             print(name)
    #             print(module)


class CSRNet_B1(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(CSRNet_B1, self).__init__()
        self.save_features = True
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.frontend = make_layers(self.frontend_feat)
        if reduce_channel == 1:
            self.backend_feat = [512, 512, 512]
            self.primary_branch = [128, 64]
        else:
            self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
            self.primary_branch = [128 // reduce_channel, 64 // reduce_channel]
        self.backend = make_layers(self.backend_feat, in_channels=512 // reduce_channel, batch_norm=True, dilation=True)

        self.output_layer = BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None')

    def forward(self, x):
        self.features = []
        features = self.frontend(x)
        x = features[-1]
        x = self.backend(x)
        x = self.output_layer(x)
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


class SSL_Model(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(SSL_Model, self).__init__()
        self.save_features = True
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        if reduce_channel == 1:
            self.backend_feat01 = [512, 512, 512]
            self.backend_feat02 = [256, 128, 64]
        else:
            self.backend_feat01 = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
            self.backend_feat02 = [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
            # [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
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
        x = self.backend(x)
        y1 = self.regressor(x)
        # y2 = self.classifier(x)
        # y2 = x.view(y2.size(0), -1)
        return y1, None

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


class CL_Model(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(CL_Model, self).__init__()
        self.save_features = True
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        if reduce_channel == 1:
            self.backend_feat01 = [512, 512, 512]
            self.backend_feat02 = [256, 128, 64]
        else:
            self.backend_feat01 = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
            self.backend_feat02 = [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
            # [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
        self.backend = make_layers(self.backend_feat01, in_channels=512 // reduce_channel, batch_norm=True,
                                   dilation=True)

        self.classifier = nn.Sequential(
            BaseConv(512 // reduce_channel, 128 // reduce_channel, kernel_size=3, bn=True),
            BaseConv(128 // reduce_channel, 64 // reduce_channel, kernel_size=3, bn=True),
            nn.AdaptiveAvgPool2d(1),
            BaseConv(64 // reduce_channel, 4, kernel_size=1, NL='None', bn=False)
        )
        self.regressor = nn.Sequential(
            make_layers(self.backend_feat02, in_channels=512 // reduce_channel, batch_norm=True, dilation=True),
            BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None', bn=False)
        )

    def forward(self, x):
        self.features = []
        features = self.frontend(x)
        x = features[-1]
        x = self.backend(x)
        y1 = self.regressor(x)
        y2 = self.classifier(x)
        y2 = x.view(y2.size(0), -1)
        return y1, y2

    def enable_features(self, flag):
        self.save_features = flag

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
            if name in ['2', '11']:
                self._modules['backend']._modules[name].register_forward_hook(get)
                print(name)
                print(module)


class Usl_Re_Model(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(Usl_Re_Model, self).__init__()
        self.save_features = True
        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        if reduce_channel == 1:
            self.backend_feat01 = [512, 512, 512]
            self.backend_feat02 = [256, 128, 64]
        else:
            self.backend_feat01 = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
            self.backend_feat02 = [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
            # [256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]
        self.backend = make_layers(self.backend_feat01, in_channels=512 // reduce_channel, batch_norm=True,
                                   dilation=True)

        # self.classifier = nn.Sequential(
        #     BaseConv(512 // reduce_channel, 128 // reduce_channel, kernel_size=3, bn=True),
        #     BaseConv(128 // reduce_channel, 64 // reduce_channel, kernel_size=3, bn=True),
        #     nn.AdaptiveAvgPool2d(1),
        #     BaseConv(64 // reduce_channel, 4, kernel_size=1, NL='None', bn=False)
        # )

        self.re_decoder = nn.Sequential(
            BaseDeConv(512 // reduce_channel, 256 // reduce_channel, kernel_size=2, stride=2, bn=True),
            BaseDeConv(256 // reduce_channel, 128 // reduce_channel, kernel_size=2, stride=2, bn=True),
            BaseDeConv(128 // reduce_channel, 64 // reduce_channel, kernel_size=2, stride=2, bn=True),
            # BaseConv(128 // reduce_channel, 64 // reduce_channel, kernel_size=3, bn=True)
            BaseConv(64 // reduce_channel, 3, kernel_size=3, NL='relu', bn=False)
        )

        self.regressor = nn.Sequential(
            make_layers(self.backend_feat02, in_channels=512 // reduce_channel, batch_norm=True, dilation=True),
            BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None', bn=False)
        )

    def forward(self, x):
        self.features = []
        size = x.size()[2:]
        features = self.frontend(x)
        x = features[-1]
        x = self.backend(x)
        y1 = self.regressor(x)
        y2 = self.re_decoder(x)

        y2 = F.interpolate(y2, size, mode='bilinear', align_corners=True)

        # y2 = x.view(y2.size(0), -1)
        return y1, y2

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


if __name__ == '__main__':
    import torchsummary

    m = CSRNet(load_weights=False)
    torchsummary.summary(m, (3, 224, 224))
