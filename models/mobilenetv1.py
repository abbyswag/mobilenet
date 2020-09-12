import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetv1(nn.Module):
    def __init__(self):
        super(MobileNetv1, self).__init__()

        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels,kernel_size= 3, stride = stride, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Linear(1024, 10)
        

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
    