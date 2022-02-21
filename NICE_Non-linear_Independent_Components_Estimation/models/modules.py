#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :modules.py
@Description  :
@Date         :2022/02/19 21:24:06
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Uniform


class CouplingLayer(nn.Module):
    """
    Implementation of the additive coupling layer from section 3.2 of the NICE paper.
    """

    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super(CouplingLayer, self).__init__()

        assert data_dim % 2 == 0

        self.mask = mask

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)

    def forward(self, x, logdet, invert=False):
        if not invert:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            z1, z2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return z1 + z2, logdet

        # Inverse additive coupling layer
        z1, z2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = z1, z2 - (self.m(z1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    """
    Implementation of the scaling layer from section 3.3 of the NICE paper.
    """

    def __init__(self, data_dim):
        super(ScalingLayer, self).__init__()

        self.log_scale_vector = nn.Parameter(
            torch.randn(1, data_dim, requires_grad=True))

    def forward(self, x, logdet, invert=False):
        log_det_jacobian = torch.sum(self.log_scale_vector)

        if invert:
            return torch.exp(-self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian


class LogisticDistribution(Distribution):
    arg_constraints = {}

    def __init__(self):
        super(LogisticDistribution, self).__init__()

    def log_prob(self, x):
        # return -F.softplus(-x)
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        if torch.cuda.is_available():
            z = Uniform(torch.cuda.FloatTensor(
                [0.]), torch.cuda.FloatTensor([1.])).sample(size)
        else:
            z = Uniform(torch.FloatTensor([0.]),
                        torch.FloatTensor([1.])).sample(size)
        z = torch.log(z) - torch.log(1. - z)
        return z.view(size)
