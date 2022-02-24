#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :modules.py
@Description  :
@Date         :2022/02/22 20:44:33
@Author       :Arctic Little Pig
@version      :1.0
'''

import numpy as np
import scipy.linalg as la
import torch
from torch import nn
from torch.nn import functional as F


def logabs(x):
    return torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channels, logdet=True):
        super(ActNorm, self).__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, x):
        # x shape: [bs, squeeze_dim, h, w]
        with torch.no_grad():
            # out shape: [squeeze_dim, bs * h * w]
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        # x shape: [bs, squeeze_dim, h, w]

        _, _, h, w = x.shape

        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)
        # out shape: [1, squeeze_dim, 1, 1]
        log_abs = logabs(self.scale)

        logdet = h * w * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (x + self.loc), logdet
        else:
            return self.scale * (x + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channels):
        super(InvConv2d, self).__init__()

        weight = torch.randn(in_channels, in_channels)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        _, _, h, w = x.shape

        out = F.conv2d(x, self.weight)
        logdet = (
            h * w *
            torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    def __init__(self, in_channels):
        super(InvConv2dLU, self).__init__()

        weight = np.random.randn(in_channels, in_channels)
        # QR decomposition of the parameter matrix, Q is the unit orthogonal matrix
        q, r = la.qr(weight)
        # print(np.allclose(weight, np.dot(q, r)))

        # LU factorization of Q matrix, P is permutation matrix, L is lower triangular matrix, U is upper triangular matrix
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        # numpy.diag returns a array view not copy, so return array is non-writable
        w_s = np.diag(w_u).copy()
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, x):
        # x shape: [bs, squeeze_dim, h, w]

        _, _, h, w = x.shape

        # out shape: [squeeze_dim, squeeze_dim, 1, 1]
        weight = self.calc_weight()
        # out shape: [bs, squeeze_dim, h, w]
        out = F.conv2d(x, weight)
        logdet = h * w * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(ZeroConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        # x shape: [bs, filter_size, h, w]

        # out shape: [bs, filter_size, h+2, w+2]
        out = F.pad(x, [1, 1, 1, 1], value=1)
        # out shape: [bs, out_channels, h, w]
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channels, filter_size=512, affine=False):
        super(AffineCoupling, self).__init__()

        self.affine = affine

        self.net = nn.Sequential(
            # out shape: [bs, filter_size, h, w]
            nn.Conv2d(in_channels // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            # out shape: [bs, filter_size, h, w]
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            # out shape: [bs, in_channels/2, h, w]
            ZeroConv2d(
                filter_size, in_channels if self.affine else in_channels // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        # x shape: [bs, squeeze_dim, h, w]

        # out shape: [bs, squeeze_dim / 2, h, w]
        in_a, in_b = x.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        else:
            # out shape: [bs, squeeze_dim / 2, h, w]
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)
