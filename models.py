import torch.nn as nn
import torch

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, infilter, midfilter, kernel_size, stride, downsample, p, flagmul):
        super().__init__()
        self.p = p
        self.flagmul = flagmul
        self.conv1 = nn.Sequential(
            nn.Conv2d(infilter, midfilter, 1, padding=0),
            nn.BatchNorm2d(midfilter),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(midfilter, midfilter, kernel_size, padding=kernel_size // 2, stride=stride),
            nn.BatchNorm2d(midfilter),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(midfilter, midfilter * self.expansion, kernel_size=1, padding=0),
            nn.BatchNorm2d(midfilter * self.expansion),
        )

        self.downsample = downsample

    def forward(self, x):
        if self.training:
            if not torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.p])):
                return x
            else:
                shortcut = x
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                if self.downsample:
                    shortcut = self.downsample(shortcut)
                return nn.ReLU()(x + shortcut)
        else:
            shortcut = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            if self.downsample != None:
                shortcut = self.downsample(shortcut)
            if self.flagmul:
                return nn.ReLU()(x * self.p + shortcut)
            else:
                return nn.ReLU()(x + shortcut)


class ResNet(nn.Module):
    def __init__(self, block, repeat, inplane=64, num_classes=10, stoc_f_l=[1, 0.5], flagmul=True):
        super().__init__()

        self.p = stoc_f_l[0]
        self.p_delta = stoc_f_l[0] - stoc_f_l[1]
        self.step = self.p_delta / (sum(repeat) - 1)
        self.flagmul = flagmul

        self.inplane = inplane
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplane, 7, padding=1, stride=2),
            nn.BatchNorm2d(self.inplane),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d((3, 3), 2)

        self.conv2 = self.make_layers(block, 64, repeat[0], 1)
        self.conv3 = self.make_layers(block, 128, repeat[1], 2)
        self.conv4 = self.make_layers(block, 256, repeat[2], 2)
        self.conv5 = self.make_layers(block, 512, repeat[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplane, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def make_layers(self, block, midfilter, repeat, stride):
        strides = [stride] + [1] * (repeat - 1)
        layers = []
        downsample = None
        for s in strides:
            if s != 1 and self.inplane != midfilter * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplane, midfilter * block.expansion, kernel_size=1, padding=0, stride=s),
                    nn.BatchNorm2d(midfilter * block.expansion)
                )
            layers.append(
                block(self.inplane, midfilter, 3, stride=s, downsample=downsample, p=self.p, flagmul=self.flagmul))
            downsample = None
            self.inplane = midfilter * block.expansion

        return nn.Sequential(*layers)
