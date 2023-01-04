import torch
from torch import nn
from torch.nn import functional as F


"""New Resdule"""
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        out = out * w  # New broadcasting feature_connection from v0.2!
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SENet(nn.Module):
    def __init__(self, block, num_blocks, input_channel):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=1)
        # self.linear = nn.Linear(512, num_classes)
        self.cnn = CNNnet()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.cnn(out)
        # print(out.shape)
        return out


# 定义网络结构 初始输入为64,用来做为decoder层
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 1, 3, 1, 1),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        return x

def SENet18(input):
    return SENet(BasicBlock, [3, 3, 3, 3], input)


# def test():
#     se_net = SENet18()
#     y = se_net(torch.randn(1, 3, 32, 32))
#     cnn = CNNnet()
#     out = cnn(y)
#     print(out.shape)
# test()


# def main():
#     # blk = ResBlk(64, 128)  # 分别给ch_in, ch_out赋值
#     # tmp = torch.randn(2, 64, 500, 500)  # 为什么有四个参数？
#     # out = blk(tmp)
#     # print('block:', out.shape)
#
#     model = ResNet18()  # num_class = 5
#     tmp = torch.randn(2, 2, 700, 700)
#
#     out = model(tmp)
#     print("resnet:", out.shape)
#     p = sum(map(lambda p:p.numel(), model.parameters()))   #其中的：lambda p:p.numel()，是什么意思？
#     print('parameters size', p)
#
#
# if __name__ == '__main__':
#     # main()
#     tmp = torch.randn(2, 2, 700, 700)
#     model = CNNnet()
#
#     print(model(tmp))