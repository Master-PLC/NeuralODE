#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :modules.py
@Description  :
@Date         :2022/02/21 16:55:34
@Author       :Arctic Little Pig
@version      :1.0
'''

from enum import IntEnum

import torch
import torch.nn as nn

from .resnet import ResNet
from .squeeze import checkerboard_mask


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """
    Description::Coupling layer in RealNVP.

    :param in_channels: Number of channels in the input.
    :param mid_channels: Number of channels in the `s` and `t` network.
    :param num_blocks: Number of residual blocks in the `s` and `t` network.
    :param mask_type: One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
    :param reverse_mask: Whether to invert the mask. Useful for alternating masks.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask=False):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask_type == MaskType.CHANNEL_WISE:
            in_channels //= 2
        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1,
                             double_after_norm=(self.mask_type == MaskType.CHECKERBOARD))

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def forward(self, x, sldj=None, invert=True):
        # x shape: [bs, c, w, h] = [256, 3, 32, 32]
        # sldj shape: [bs] = [256]
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            # out shape: [1, 1, w, h] = [1, 1, 32, 32]
            b = checkerboard_mask(x.size(2), x.size(
                3), self.reverse_mask, device=x.device)
            # out shape: [bs, c, w, h] = [256, 3, 32, 32]
            x_b = x * b
            # out shape: [bs, 2c, w, h] = [256, 6, 32, 32]
            st = self.st_net(x_b)
            # out shape: [bs, c, w, h], [bs, c, w, h] = [256, 3, 32, 32], [256, 3, 32, 32]
            s, t = st.chunk(2, dim=1)
            # out shape: [bs, c, w, h] = [256, 3, 32, 32]
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            # Scale and translate
            if invert:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                # out shape: [256, 6, 16, 16], [256, 6, 16, 16]
                x_change, x_id = x.chunk(2, dim=1)
            # out shape: [256, 12, 16, 16]
            st = self.st_net(x_id)
            # out shape: [256, 6, 16, 16],[256, 6, 16, 16]
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))

            # Scale and translate
            if invert:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                # out shape: [256, 12, 16, 16]
                x = torch.cat((x_change, x_id), dim=1)
        return x, sldj


class Rescale(nn.Module):
    """
    Description::Per-channel rescaling. Need a proper `nn.Module` so we can wrap it with `torch.nn.utils.weight_norm`.

    :param num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()

        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        # out shape: [bs, c, w, h] = [256, 3, 32, 32]
        x = self.weight * x
        return x
