import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1,expansion = 6):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, 
                                  1,1,0, bias=False)
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels * expansion, 
                                  kernel_size=3, stride=stride, padding=1, groups = in_channels * expansion, bias=False)
        self.conv3 = nn.Conv2d(in_channels * expansion, out_channels, 1,1,0, bias=False)
        self.relu6 = nn.ReLU6(inplace = True)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion)
        self.bn2 = nn.BatchNorm2d(in_channels * expansion)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.expansion = expansion
        self.shortcut =  stride == 1 and in_channels == out_channels
            
    def forward(self, X):
            if self.expansion==1:
                output = self.relu6(self.bn2(self.conv2(X))) #dw
                output = self.bn3(self.conv3(output)) #pw
            else:
                output = self.conv1(X)
                output = self.relu6(self.bn1(output)) #pw
                output = self.relu6(self.bn2(self.conv2(output))) #dw
                output = self.bn3(self.conv3(output)) #pw
                
            if self.shortcut:
                output += X
            return output

class MobileNetV2(nn.Module):
    def __init__(self, cfg, n_class=10):
        super(MobileNetV2, self).__init__()
        self.input_channel = 32
        self.out_channel = 1280
        self.cfg = cfg  

        self.features = [
        nn.Conv2d(3, self.input_channel, 3, 2, 1),
        nn.BatchNorm2d(self.input_channel),
        nn.ReLU6(inplace=True)]
       
        for t, c, n, s in self.cfg:
            for i in range(n):
                if i == 0:
                    self.features.append(Bottleneck(self.input_channel, c, s, t))
                else:
                    self.features.append(Bottleneck(self.input_channel, c, 1, t))
                self.input_channel =c
        self.features = nn.Sequential(*self.features)
        self.pw = nn.Sequential(
        nn.Conv2d(self.input_channel, self.out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(self.out_channel),
        nn.ReLU6(inplace=True)
    )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.out_channel, n_class)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(self.pw(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

        