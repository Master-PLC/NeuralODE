#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :resnet.py
@Description  :
@Date         :2022/02/12 16:39:23
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adjoint import OdeWithGrad


def norm(dim, type='group_norm'):
    if type == 'group_norm':
        return nn.GroupNorm(min(32, dim), dim)

    return nn.BatchNorm2d(dim)


def conv3x3(in_feats, out_feats, stride=1):
    return nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1, bias=False)


def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape

    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


class ConvBlockOde(OdeWithGrad):
    def __init__(self, dim=64):
        super(ConvBlockOde, self).__init__()
        # 1 additional dim for time

        # out shape: [bs, 64, 6, 6]
        self.conv1 = conv3x3(dim + 1, dim)
        # out shape: [bs, 64, 6, 6]
        self.norm1 = norm(dim)
        # out shape: [bs, 64, 6, 6]
        self.conv2 = conv3x3(dim + 1, dim)
        # out shape: [bs, 64, 6, 6]
        self.norm2 = norm(dim)

    def forward(self, x, t):
        # x shape: [bs, 64, 6, 6]
        # t shape: [2]

        # out shape: [bs, 64 + 1, 6, 6] = [bs, 65, 6, 6]
        xt = add_time(x, t)
        # out shape: [bs, 64, 6, 6]
        h = self.norm1(torch.relu(self.conv1(xt)))
        # out shape: [bs, 64 + 1, 6, 6] = [bs, 65, 6, 6]
        ht = add_time(h, t)
        # out shape: [bs, 64, 6, 6]
        dxdt = self.norm2(torch.relu(self.conv2(ht)))

        return dxdt


class ResBlock(nn.Module):
    # y0 shape: [bs, 64, 6, 6]
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class ContinuousResNet(nn.Module):
    def __init__(self, feature, channels=1):
        super(ContinuousResNet, self).__init__()
        self.downsampling = nn.Sequential(
            # out shape: [bs, 64, 26, 26]
            nn.Conv2d(channels, 64, 3, 1),
            # out shape: [bs, 64, 26, 26]
            norm(64),
            nn.ReLU(inplace=True),
            # out shape: [bs, 64, 13, 13]
            nn.Conv2d(64, 64, 4, 2, 1),
            # out shape: [bs, 64, 13, 13]
            norm(64),
            nn.ReLU(inplace=True),
            # out shape: [bs, 64, 6, 6]
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        # out shape: [bs, 64, 6, 6]
        self.feature = feature
        # out shape: [bs, 64, 6, 6]
        self.norm = norm(64)
        # out shape: [bs, 64, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # out shape: [bs, 10]
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # x shape: [bs, c, w, h] = [128, 1, 28, 28]
        # out shape: [bs, 64, 6, 6]
        x = self.downsampling(x)
        # out shape: [bs, 64, 6, 6]
        x = self.feature(x)
        x = self.norm(x)
        x = F.relu(x)
        # out shape: [bs, 64, 1, 1]
        x = self.avg_pool(x)
        # shape: 64 * 1 * 1 = 64
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        # out shape: [bs, shape] = [bs, 64]
        x = x.view(-1, shape)
        # out shape: [bs, 10]
        out = self.fc(x)

        return out
