import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def h_sigmoid(x):
    return F.relu6(x + 3, inplace=True) / 6

class SE(nn.Module):
    def __init__(self, in_dim, reduction=4):
        super(SE, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim // reduction, 1,1,0, bias = False)
        self.conv2 = nn.Conv2d(in_dim // reduction, in_dim, 1,1,0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dim // reduction)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.hswish = hswish()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(self.avgpool(x))))
        x = self.hswish(self.bn2(self.conv2(x)))
        return x

class hswish(nn.Module):
    def forward(self, x):
        out = x *h_sigmoid(x)
        return out

class BottleneckSE(nn.Module): 
    def __init__(self, in_channels, hid_channels, out_channels, kernel, stride,  se, hs):
        super(BottleneckSE, self).__init__()
        self.se = se
        self.stride = stride 
        self.conv1 = nn.Conv2d(in_channels, hid_channels, 
                                  1,1,0, bias=False)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, 
                                  kernel_size=kernel, stride=stride,  padding= kernel//2, groups = hid_channels, bias=False)
        self.conv3 = nn.Conv2d(hid_channels, out_channels, 1,1,0, bias=False)
        self.hs = hs
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

            
    def forward(self, X):
            output = self.hs(self.bn1(self.conv1(X)))
            output = self.hs(self.bn2(self.conv2(output)))
            output = self.bn3(self.conv3(output))
            if self.se !=None:
                output = self.se(output)
            if self.stride ==1:
                output += self.shortcut(X)
            return output


class MobileNetV3(nn.Module):
    def __init__(self, cfg, n_class=10):
        super(MobileNetV3, self).__init__()
        self.input_channel = 16
        self.cfg = cfg     
        self.hswish = hswish()
        self.features = [
        nn.Conv2d(3,  self.input_channel, 3, 2, 1, bias=False),
        nn.BatchNorm2d( self.input_channel),
        self.hswish ]
        self.relu = nn.ReLU(inplace=True)
        
        for k, t, c, se, hs, s in self.cfg:
            if se:
                if hs == 'RE':
                    self.features.append(BottleneckSE(self.input_channel, t,c,k, s, SE(c), self.relu))
                else:
                    self.features.append(BottleneckSE(self.input_channel, t,c,k, s, SE(c), self.hswish))
            else:
                if hs == 'RE':
                    self.features.append(BottleneckSE(self.input_channel, t,c,k, s, None, self.relu))
                else:
                    self.features.append(BottleneckSE(self.input_channel, t,c,k, s, None, self.hswish))
            self.input_channel = c
        self.features = nn.Sequential(*self.features)
        self.pw = nn.Sequential(
        nn.Conv2d(self.input_channel, t, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(t),
        self.hswish
    )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(t, 1280)
        self.bn1 = nn.BatchNorm1d(1280)
        self.linear2 = nn.Linear(1280, n_class)

    def forward(self, x):
        
        x = self.features(x)
        x = self.avgpool(self.pw(x))
        x = x.view(x.size(0), -1)
        x = self.linear2(self.hswish(self.bn1(self.linear1(x))))
        return x

