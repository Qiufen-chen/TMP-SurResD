# Author QFIUNE
# coding=utf-8
# @Time: 2022/2/24 17:24
# @File: resnet.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Define residual blocks
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.shortcut(x)
        out = F.elu(out)
        return out

# ----------------------------------------------------------------------------------
class ResNet_2D(nn.Module):
    def __init__(self, block, num_blocks, input_channel):
        super(ResNet_2D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        purpose: Duplicate layer
        :param block: BasicBlock
        :param planes:
        :param num_blocks:
        :param stride:
        :return:
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        drop = nn.Dropout2d(p=0.5, inplace=False)

        out = drop(F.elu(self.conv2(out)))
        out = F.elu(self.conv3(out))
        out = F.elu(self.conv4(out))
        return out

# ----------------------------------------------------------------------------------
def ResNet18(n):
    return ResNet_2D(BasicBlock, [2, 2, 2, 2], n)


if __name__ == '__main__':
    model = ResNet18()
    tmp = torch.rand(2, 10, 5, 5)
    y = model(tmp)
    print(y.shape)
