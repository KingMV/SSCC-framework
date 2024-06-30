# import torch
import torch.nn as nn
# from torch.utils import model_zoo
from torchvision import models


class resnet(nn.Module):
    def __init__(self, pretrained=False):
        super(resnet, self).__init__()
        res = models.resnet50(pretrained=pretrained)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.inplanes = 512
        self.own_layer_3 = self.make_res_layer(Bottleneck, 256, 6, stride=1)
        self.own_layer_4 = self.make_res_layer(Bottleneck, 512, 3, stride=1)

        self.own_layer_3.load_state_dict(res.layer3.state_dict())
        self.own_layer_4.load_state_dict(res.layer4.state_dict())

    def forward(self, input):
        # print(input)
        input = self.frontend(input)
        input = self.own_layer_3(input)
        input = self.own_layer_4(input)
        return input

    def make_res_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    import torchsummary

    m = resnet(pretrained=False)
    torchsummary.summary(m, (3, 224, 224))
