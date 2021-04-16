import torch.nn as nn
import torch
import torchvision
from torchvision import datasets, transforms
from models import ResNet, Bottleneck

"""
    stoc_f_l : [첫번째 레이어의 survival probability, 마지막 레이어의 survival probability]
    flagmul : test 시 survival probability를 고려 할 것인가.
"""

resnet = ResNet(Bottleneck, [3,4,6,3],inplane = 64, num_classes=10,stoc_f_l = [1, 0.5], flagmul = True)

print(resnet)