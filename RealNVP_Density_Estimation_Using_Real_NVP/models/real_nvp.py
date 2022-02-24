#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :real_nvp.py
@Description  :
@Date         :2022/02/21 17:01:35
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CouplingLayer, MaskType
from .squeeze import squeeze_2x2


class RealNVP(nn.Module):
    """
    Description::RealNVP Model

    :param num_scales: Number of scales in the RealNVP model.
    :param in_channels: Number of channels in the input.
    :param mid_channels: Number of channels in the intermediate layers.
    :param num_blocks: Number of residual blocks in the s and t network of Coupling layers.
    """

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor(
            [0.9], dtype=torch.float32))

        self.flows = _RealNVP(0, num_scales, in_channels,
                              mid_channels, num_blocks)

    def forward(self, x, invert=False):
        # x shape: [bs, c, w, h] = [256, 3, 32, 32]
        sldj = None
        if not invert:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError(
                    f'Expected x in [0, 1], got x with min/max {x.min()}/{x.max()}')

            # De-quantize and convert to logits
            # out shape: [bs, c, w, h], [bs]
            x, sldj = self._pre_process(x)

        x, sldj = self.flows(x, sldj, invert)
        return x, sldj

    def _pre_process(self, x):
        """
        Description::Dequantize the input image `x` and convert to logits.

        :param x: Input image.

        :returns y: Dequantized logits of `x`.

        Usage::
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """

        # Quantization, add uniform noise and dequantization
        y = (x * 255. + torch.rand_like(x)) / 256.
        # constrain y to [-0.9, 0.9]
        y = (2 * y - 1) * self.data_constraint
        # transform y to [0.05, 0.95]
        y = (y + 1) / 2
        # TODO: why?
        y = y.log() - (1. - y).log()

        # Sum log-determinant of Jacobian of initial transform
        # ldj shape: [bs, c, w, h] = [256, 3, 32, 32]
        ldj = F.softplus(y) + F.softplus(-y) - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
        # sldj shape: [bs] = [256]
        sldj = ldj.view(ldj.size(0), -1).sum(-1)
        return y, sldj


class _RealNVP(nn.Module):
    """
    Description::Recursive builder for a RealNVP model.

    :param scale_idx: Index of current scale.
    :param num_scales: Number of scales in the RealNVP model.
    :param in_channels: Number of channels in the input.
    :param mid_channels: Number of channels in the intermediate layers.
    :param num_blocks: Number of residual blocks in the s and t network of Coupling layers.
    """

    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = (scale_idx == num_scales - 1)

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks,
                          MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks,
                          MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks,
                          MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels,
                              num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels,
                              num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels,
                              num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = _RealNVP(
                scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, invert=False):
        # x shape: [bs, c, w, h] = [256, 3, 32, 32]
        # sldj shape: [bs] = [256]
        if invert:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, invert=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, invert)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, invert=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, invert=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, invert)
                x = squeeze_2x2(x, invert=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, invert)
        else:
            for coupling in self.in_couplings:
                # out shape: [bs, c, w, h], [bs] = [256, 3, 32, 32], [256]
                x, sldj = coupling(x, sldj, invert)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                # out shape: [256, 12, 16, 16]
                x = squeeze_2x2(x, invert=False)
                for coupling in self.out_couplings:
                    # out shape: [bs, 4c, w/2, h/2], [bs] = [256, 12, 16, 16], [256]
                    x, sldj = coupling(x, sldj, invert)
                # out shape: [256, 3, 32, 32]
                x = squeeze_2x2(x, invert=True)

                # Re-squeeze -> split -> next block
                # out shape: [bs, 4c, w/2, h/2] = [256, 12, 16, 16]
                x = squeeze_2x2(x, invert=False, alt_order=True)
                # out shape: [256, 6, 16, 16], [256, 6, 16, 16]
                x, x_split = x.chunk(2, dim=1)
                # out shape: [bs, 2c, w/2, h/2], [bs] = [256, 6, 16, 16], [256]
                x, sldj = self.next_block(x, sldj, invert)
                # out shape: [bs, 4c, w/2, h/2] = [256, 12, 16, 16]
                x = torch.cat((x, x_split), dim=1)
                # out shape: [bs, c, w, h] = [256, 3, 32, 32]
                x = squeeze_2x2(x, invert=True, alt_order=True)
        return x, sldj
