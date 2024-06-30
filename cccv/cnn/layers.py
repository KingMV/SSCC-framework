import torch
import torch.nn as nn


# class BaseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', activation=None, bn=False):
#         super(BaseConv, self).__init__()
#         self.use_bn = bn
#         self.activation = activation
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)
#         self.conv.weight.data.normal_(0, 0.01)
#         self.conv.bias.data.zero_()
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.bn.weight.data.fill_(1)
#         self.bn.bias.data.zero_()
#
#     def forward(self, input):
#         input = self.conv(input)
#         if self.use_bn:
#             input = self.bn(input)
#         if self.activation:
#             input = self.activation(input)
#
#         return input


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_drop=False, NL='relu', same_padding=True,
                 bn=False,
                 norm_layer=None,
                 dilation=1):
        super(BaseConv, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.bn = norm_layer(out_channels) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        elif NL == 'leaky_relu':
            self.relu = nn.LeakyReLU()
        else:
            self.relu = None
        if use_drop:
            self.drop = nn.Dropout2d(p=0.5)
        else:
            self.drop = None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


class BaseDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=True, bn=False):
        super(BaseDeConv, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels, affine=True) if bn else None
        # self.bn = nn.InstanceNorm2d(out_channels, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
