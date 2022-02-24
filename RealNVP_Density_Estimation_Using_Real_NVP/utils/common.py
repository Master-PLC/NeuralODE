#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :common.py
@Description  :
@Date         :2022/02/21 16:57:37
@Author       :Arctic Little Pig
@version      :1.0
'''

import numpy as np
import torch.nn.utils as utils


class AverageMeter(object):
    """
    Description::Computes and stores the average and current value.

    Usage::
        - Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """

    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def bits_per_dim(x, nll):
    """
    Description::Get the bits per dimension implied by using model with `loss`
        for compressing `x`, assuming each entry can take on `k` discrete values.

    :param x: Input to the model. Just used for dimensions.
    :param nll: Scalar negative log-likelihood loss tensor.

    :returns bpd: Bits per dimension implied if compressing `x`.
    """

    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """
    Description::Clip the norm of the gradients for all parameters under `optimizer`.

    :param optimizer: torch.optim.Optimizer
    :param max_norm: The maximum allowable norm of gradients.
    :param norm_type: The type of norm to use in computing gradient norms.
    """

    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)
