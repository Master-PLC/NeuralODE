#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :loss.py
@Description  :
@Date         :2022/02/21 17:00:17
@Author       :Arctic Little Pig
@version      :1.0
'''

import numpy as np
import torch.nn as nn


class RealNVPLoss(nn.Module):
    """
    Description::Get the NLL loss for a RealNVP model.

    :param k: Number of discrete values in each input dimension.
        E.g., `k` is 256 for natural images.

    Usage::
        - Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()

        self.k = k

    def forward(self, z, sldj):
        # p(z) is standard normal distribution
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) - \
            np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll
