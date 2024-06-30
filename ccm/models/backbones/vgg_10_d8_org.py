from torch.utils import model_zoo
import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class VGG13_BN(nn.Module):
    def __init__(self):
        super(VGG13_BN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        # input = self.pool(conv4_3)
        # input = self.conv5_1(input)
        # input = self.conv5_2(input)
        # conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3


def load_vgg13_weight():
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


def vgg_block(pretrained=True):
    net = VGG13_BN()
    if pretrained:
        net.load_state_dict(load_vgg13_weight())
    return net
