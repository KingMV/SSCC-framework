import torch.nn as nn
from cccv.cnn.layers import BaseConv
from torch.utils import model_zoo
from ccm.models.CCNet.custom_BN import MixBatchNorm2d, SplitBatchNorm2d, DSBN

from config.sys_config_back import sys_cfg

if sys_cfg.bn_dtype == 'mix':
    norm_layer = MixBatchNorm2d
elif sys_cfg.bn_dtype == 'split':
    norm_layer = SplitBatchNorm2d
elif sys_cfg.bn_dtype == 'dsbn':
    norm_layer = DSBN
else:
    norm_layer = nn.BatchNorm2d

from torchvision.models import vgg


# class VGG13_BN(nn.Module):
#     def __init__(self):
#         super(VGG13_BN, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv1_1 = BaseConv(3, 64, 3, 1, NL='relu', bn=True)
#         self.conv1_2 = BaseConv(64, 64, 3, 1, NL='relu', bn=True)
#         self.conv2_1 = BaseConv(64, 128, 3, 1, NL='relu', bn=True)
#         self.conv2_2 = BaseConv(128, 128, 3, 1, NL='relu', bn=True)
#         self.conv3_1 = BaseConv(128, 256, 3, 1, NL='relu', bn=True)
#         self.conv3_2 = BaseConv(256, 256, 3, 1, NL='relu', bn=True)
#         self.conv3_3 = BaseConv(256, 256, 3, 1, NL='relu', bn=True)
#         self.conv4_1 = BaseConv(256, 512, 3, 1, NL='relu', bn=True)
#         self.conv4_2 = BaseConv(512, 512, 3, 1, NL='relu', bn=True)
#         self.conv4_3 = BaseConv(512, 512, 3, 1, NL='relu', bn=True)
#         # self.conv5_1 = BaseConv(512, 512, 3, 1, NL='relu', bn=True)
#         # self.conv5_2 = BaseConv(512, 512, 3, 1, NL='relu', bn=True)
#         # self.conv5_3 = BaseConv(512, 512, 3, 1, NL='relu', bn=True)
#
#     def forward(self, input):
#         # print(input.shape)
#         input = self.conv1_1(input)
#         input = self.conv1_2(input)
#         input = self.pool(input)
#         input = self.conv2_1(input)
#         conv2_2 = self.conv2_2(input)
#
#         input = self.pool(conv2_2)
#         input = self.conv3_1(input)
#         input = self.conv3_2(input)
#         conv3_3 = self.conv3_3(input)
#
#         input = self.pool(conv3_3)
#         input = self.conv4_1(input)
#         input = self.conv4_2(input)
#         conv4_3 = self.conv4_3(input)
#
#         # input = self.pool(conv4_3)
#         # input = self.conv5_1(input)
#         # input = self.conv5_2(input)
#         # conv5_3 = self.conv5_3(input)
#
#         return conv2_2, conv3_3, conv4_3

class VGG13(nn.Module):
    def __init__(self, reduce_channel=1, use_drop=False):
        super(VGG13, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64 // reduce_channel, 3, 1, NL='relu', bn=False, norm_layer=norm_layer)
        self.conv1_2 = BaseConv(64 // reduce_channel, 64 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv2_1 = BaseConv(64 // reduce_channel, 128 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv2_2 = BaseConv(128 // reduce_channel, 128 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv3_1 = BaseConv(128 // reduce_channel, 256 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv3_2 = BaseConv(256 // reduce_channel, 256 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv3_3 = BaseConv(256 // reduce_channel, 256 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv4_1 = BaseConv(256 // reduce_channel, 512 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv4_2 = BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv4_3 = BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, NL='relu', bn=False,
                                norm_layer=norm_layer, use_drop=use_drop)

    def forward(self, input):
        # print(input.shape)
        input = self.conv1_1(input)
        input = self.conv1_2(input)

        input = self.pool(input)
        input = self.conv2_1(input)
        input = self.conv2_2(input)

        input = self.pool(input)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        input = self.conv3_3(input)

        input = self.pool(input)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        input = self.conv4_3(input)

        return input


class VGG13_BN(nn.Module):
    def __init__(self, reduce_channel=1, use_drop=False):
        super(VGG13_BN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64 // reduce_channel, 3, 1, NL='relu', bn=True, norm_layer=norm_layer)
        self.conv1_2 = BaseConv(64 // reduce_channel, 64 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv2_1 = BaseConv(64 // reduce_channel, 128 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv2_2 = BaseConv(128 // reduce_channel, 128 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv3_1 = BaseConv(128 // reduce_channel, 256 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv3_2 = BaseConv(256 // reduce_channel, 256 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv3_3 = BaseConv(256 // reduce_channel, 256 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv4_1 = BaseConv(256 // reduce_channel, 512 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv4_2 = BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)
        self.conv4_3 = BaseConv(512 // reduce_channel, 512 // reduce_channel, 3, 1, NL='relu', bn=True,
                                norm_layer=norm_layer, use_drop=use_drop)

    def forward(self, input):
        # print(input.shape)
        input = self.conv1_1(input)
        input = self.conv1_2(input)

        input = self.pool(input)
        input = self.conv2_1(input)
        input = self.conv2_2(input)

        input = self.pool(input)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        input = self.conv3_3(input)

        input = self.pool(input)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        input = self.conv4_3(input)

        return input


def load_vgg13bn_weight():
    state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
    old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37]
    new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3']
    new_dict = {}

    for i in range(10):
        new_dict['conv' + new_name[i] + '.conv.weight'] = \
            state_dict['features.' + str(old_name[2 * i]) + '.weight']
        new_dict['conv' + new_name[i] + '.conv.bias'] = \
            state_dict['features.' + str(old_name[2 * i]) + '.bias']
        new_dict['conv' + new_name[i] + '.bn.weight'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
        new_dict['conv' + new_name[i] + '.bn.bias'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
        new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
        new_dict['conv' + new_name[i] + '.bn.running_var'] = \
            state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']
    return new_dict


def load_vgg13_weight():
    state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
    old_name = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3']
    new_dict = {}

    for i in range(10):
        new_dict['conv' + new_name[i] + '.conv.weight'] = \
            state_dict['features.' + str(old_name[i]) + '.weight']
        new_dict['conv' + new_name[i] + '.conv.bias'] = \
            state_dict['features.' + str(old_name[i]) + '.bias']
        # new_dict['conv' + new_name[i] + '.bn.weight'] = \
        #     state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
        # new_dict['conv' + new_name[i] + '.bn.bias'] = \
        #     state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
        # new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
        #     state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
        # new_dict['conv' + new_name[i] + '.bn.running_var'] = \
        #     state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']
    return new_dict


def vgg_block(pretrained=True, use_drop=False, bn=True, rc=1):
    if bn:
        net = VGG13_BN(rc, use_drop)
        if pretrained and rc == 1:
            net.load_state_dict(load_vgg13bn_weight(), strict=False)
    else:
        net = VGG13(rc, use_drop)
        if pretrained and rc == 1:
            net.load_state_dict(load_vgg13_weight(), strict=False)

    return net

# def vgg_block(pretrained=True, use_drop=False, rc=1):
#     CCNet = VGG13_BN(rc, use_drop)
#     if pretrained and rc == 1:
#         CCNet.load_state_dict(load_vgg13_weight(), strict=False)
#     return CCNet
# if __name__ == '__main__':
#     conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
