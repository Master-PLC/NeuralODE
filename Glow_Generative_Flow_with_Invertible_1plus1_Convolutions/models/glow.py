#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :glow.py
@Description  :
@Date         :2022/02/22 20:42:03
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn as nn
from utils.common import gaussian_log_p, gaussian_sample

from .modules import ActNorm, AffineCoupling, InvConv2d, InvConv2dLU, ZeroConv2d


class Glow(nn.Module):
    def __init__(self, in_channels, num_flow, num_block, affine=False, conv_lu=True):
        super(Glow, self).__init__()

        self.blocks = nn.ModuleList()
        num_channels = in_channels
        for _ in range(num_block - 1):
            self.blocks.append(Block(num_channels, num_flow,
                               affine=affine, conv_lu=conv_lu))
            num_channels *= 2
        self.blocks.append(Block(num_channels, num_flow,
                           split=False, affine=affine))

    def forward(self, x):
        # x shape: [bs, c, h, w] = [256, 3, 64, 64]
        log_p_sum = 0
        logdet = 0
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(
                    z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                x = block.reverse(
                    x, z_list[-(i + 1)], reconstruct=reconstruct)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, num_flow, split=True, affine=False, conv_lu=True):
        super(Block, self).__init__()

        squeeze_dim = in_channels * 4

        self.flows = nn.ModuleList()
        for _ in range(num_flow):
            self.flows.append(
                Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channels * 2, in_channels * 4)
        else:
            self.prior = ZeroConv2d(in_channels * 4, in_channels * 8)

    def forward(self, x):
        # x shape: [bs, c, h, w]

        bs, c, h, w = x.shape
        # out shape: [bs, c, h // 2, 2, w // 2, 2]
        squeezed = x.view(bs, c, h // 2, 2, w // 2, 2)
        # out shape: [bs, c, 2, 2, h // 2, w // 2]
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        # out shape: [bs, c * 4, h // 2, w // 2]
        out = squeezed.contiguous().view(bs, c * 4, h // 2, w // 2)

        logdet = 0
        for flow in self.flows:
            # out shape: [bs, c * 4, h // 2, w // 2]
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            # out shape: [bs, c * 2, h // 2, w // 2]
            out, z_new = out.chunk(2, 1)
            # out shape: [bs, c * 2, h // 2, w // 2]
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(bs, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(bs, -1).sum(1)
            z_new = out
        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        x = output

        if reconstruct:
            if self.split:
                x = torch.cat([output, eps], 1)
            else:
                x = eps
        else:
            if self.split:
                mean, log_sd = self.prior(x).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(x)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = z

        for flow in self.flows[::-1]:
            x = flow.reverse(x)

        bs, c, h, w = x.shape

        unsqueezed = x.view(bs, c // 4, 2, 2, h, w)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(bs, c // 4, h * 2, w * 2)
        return unsqueezed


class Flow(nn.Module):
    def __init__(self, in_channels, affine=False, conv_lu=True):
        super(Flow, self).__init__()

        self.actnorm = ActNorm(in_channels)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channels)
        else:
            self.invconv = InvConv2d(in_channels)

        self.coupling = AffineCoupling(in_channels, affine=affine)

    def forward(self, x):
        # x shape: [bs, squeeze_dim, h, w]
        
        # out shape: [bs, squeeze_dim, h, w]
        out, logdet = self.actnorm(x)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        x = self.coupling.reverse(output)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)
        return x
