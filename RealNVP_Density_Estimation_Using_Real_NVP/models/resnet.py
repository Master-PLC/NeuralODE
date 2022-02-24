#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :resnet.py
@Description  :
@Date         :2022/02/21 20:11:28
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import WNConv2d


class ResNet(nn.Module):
    """
    Description::ResNet for scale and translate factors in Real NVP.

    :param in_channels: Number of channels in the input.
    :param mid_channels: Number of channels in the intermediate layers.
    :param out_channels: Number of channels in the output.
    :param num_blocks: Number of residual blocks in the network.
    :param kernel_size: Side length of each filter in convolutional layers.
    :param padding: Padding for convolutional layers.
    :param double_after_norm: Double input after input BatchNorm.
    """

    def __init__(self, in_channels, mid_channels, out_channels, num_blocks, kernel_size=3, padding=1, double_after_norm=True):
        super(ResNet, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        # If mask_type == MaskType.CHECKERBOARD, double the input
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(
            2 * in_channels, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(
            mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(
            mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # x shape: [bs, c, w, h] = [256, 3, 32, 32]

        # out shape: [bs, c, w, h] = [256, 3, 32, 32]
        x = self.in_norm(x)
        if self.double_after_norm:
            x *= 2.
        # out shape: [bs, 2c, w, h] = [256, 6, 32, 32]
        x = torch.cat((x, -x), dim=1)
        x = F.relu(x)
        # out shape: [bs, mc, w, h] = [256, 64, 32, 32]
        x = self.in_conv(x)
        # out shape: [bs, mc, w, h] = [256, 64, 32, 32]
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            # out shape: [bs, mc, w, h] = [256, 64, 32, 32]
            x = block(x)
            # out shape: [bs, mc, w, h] = [256, 64, 32, 32]
            x_skip += skip(x)
        # out shape: [bs, mc, w, h] = [256, 64, 32, 32]
        x = self.out_norm(x_skip)
        x = F.relu(x)
        # out shape: [bs, 2c, w, h] = [256, 6, 32, 32]
        x = self.out_conv(x)
        return x


class ResidualBlock(nn.Module):
    """
    Description::ResNet basic block with weight norm.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels,
                                kernel_size=3, padding=1, bias=False)

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # x shape: [bs, mc, w, h] = [256, 64, 32, 32]

        skip = x
        # out shape: [bs, mc, w, h] = [256, 64, 32, 32]
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = x + skip
        return x
