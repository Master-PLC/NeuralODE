#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :nice.py
@Description  :
@Date         :2022/02/19 18:29:26
@Author       :Arctic Little Pig
@version      :1.0
'''

import numpy as np
import torch
import torch.nn as nn

from .modules import CouplingLayer, LogisticDistribution, ScalingLayer


class NICE(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_coupling_layers=3, num_net_layers=6):
        super(NICE, self).__init__()

        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # alternating mask orientations for consecutive coupling layers
        # 奇数层选择偶数序列
        masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                 for i in range(num_coupling_layers)]
        # print(masks)

        self.coupling_layers = nn.ModuleList([CouplingLayer(
            data_dim=data_dim, hidden_dim=hidden_dim, mask=masks[i], num_layers=num_net_layers) for i in range(num_coupling_layers)])

        self.scaling_layer = ScalingLayer(data_dim=data_dim)

        self.prior = LogisticDistribution()

    def forward(self, x, invert=False):
        if not invert:
            z, log_det_jacobian = self.f(x)
            log_likelihood = torch.sum(
                self.prior.log_prob(z), dim=1) + log_det_jacobian
            return z, log_likelihood
        return self.f_inverse(x)

    def f(self, x):
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(self.coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def f_inverse(self, z):
        x = z
        x, _ = self.scaling_layer(x, 0, invert=True)
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, self.data_dim]).view(
            self.samples, self.data_dim)
        return self.f_inverse(z)

    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask  # flip mask orientation
        mask = torch.tensor(mask)
        mask = mask.cuda()
        return mask.float()
