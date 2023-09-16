import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        for i in self.bn3.parameters():
            i.requires_grad = False

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        # downsample if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, paddings_list, dilation_list):
        super().__init__()
        self.module_list = nn.ModuleList()
        for padding, dilation in zip(paddings_list, dilation_list):
            self.module_list.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
            )

        for m in self.module_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.module_list[0](x)
        for i in range(len(self.module_list) - 1):
            out += self.module_list[i + 1](x)
            return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        for i in self.bn1.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)  # conv2
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)  # conv3
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=1,
                                       dilation=2)  # atrous convolution, conv4
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=1,
                                       dilation=4)  # atrous convolution, conv5
        self.layer5 = self._make_pred_layer(ASPP, 2048, num_classes, [6, 12, 18, 24], [6, 12, 18, 24])  # aspp

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # number of parameters
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, ResBlock, blocks, planes, stride=1, dilation=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers.append(ResBlock(self.in_channels, planes, downsample=downsample, stride=stride, dilation=dilation))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, ResBlock, in_ch, out_ch, paddings_list, dilation_list):
        return ResBlock(in_ch, out_ch, paddings_list, dilation_list)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer.
        """
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [self.layer5.parameters()]                 #The last layer has 10 times the learning rate of other layers.

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def DeepLabv2(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)
