import torch.nn as nn
from ccm.models.backbones.vgg_10_d8 import vgg_block
# from ccm.models.backbones.vgg_10_d8_org import vgg_block
from cccv.cnn.layers import BaseConv


class VGG_RC(nn.Module):
    def __init__(self, load_weights=False, reduce_channel=1):
        super(VGG_RC, self).__init__()
        # self.frontend = vgg_block(pretrained=load_weights)

        self.frontend = vgg_block(pretrained=load_weights, rc=reduce_channel)
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.frontend = make_layers(self.frontend_feat)
        # if reduce_channel == 1:
        #     self.backend_feat = [512, 512, 512, 256, 128, 64]
        # else:
        #     self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel,
        #                          256 // reduce_channel, 128 // reduce_channel, 64 // reduce_channel]

        if reduce_channel == 1:
            self.backend_feat = [512, 512, 512]
        else:
            self.backend_feat = [512 // reduce_channel, 512 // reduce_channel, 512 // reduce_channel]
        self.backend = make_layers(self.backend_feat, in_channels=512 // reduce_channel, batch_norm=True, dilation=True)
        # self.output_layer = BaseConv(64 // reduce_channel, 1, kernel_size=1, NL='None')
        self.classifier = nn.Sequential(
            BaseConv(512 // reduce_channel, 128 // reduce_channel, kernel_size=3, bn=True),
            BaseConv(128 // reduce_channel, 64 // reduce_channel, kernel_size=3, bn=True))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = BaseConv(64 // reduce_channel, 4, kernel_size=1, NL='None', bn=False)
        # self.fc_layer = BaseConv(64 // reduce_channel, 4, kernel_size=1, NL='None')

    def forward(self, x):
        self.features = []
        features = self.frontend(x)
        x = features[-1]
        x = self.backend(x)
        x = self.classifier(x)
        x = self.avg(x)
        x = self.fc_layer(x)
        x = x.view(x.size(0), -1)
        return x

    def get(self, model, input, output):
        # function will be automatically called each time, since the hook is injected
        self.features.append(output.detach())

    def regist_hook(self):
        self.features = []

        def get(model, input, output):
            # function will be automatically called each time, since the hook is injected
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


if __name__ == '__main__':
    import torchsummary

    m = VGG_RC(load_weights=False)
    torchsummary.summary(m, (3, 224, 224))
