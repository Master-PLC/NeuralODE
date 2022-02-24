#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :norm.py
@Description  :
@Date         :2022/02/21 16:59:50
@Author       :Arctic Little Pig
@version      :1.0
'''

import functools

import torch
import torch.nn as nn


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            'Invalid normalization type: {}'.format(norm_type))


def get_param_groups(net, weight_decay, norm_suffix='weight_g', verbose=False):
    """Get two parameter groups from `net`: One named "normalized" which will
    override the optimizer with `weight_decay`, and one named "unnormalized"
    which will inherit all hyperparameters from the optimizer.

    Args:
        net (torch.nn.Module): Network to get parameters from
        weight_decay (float): Weight decay to apply to normalized weights.
        norm_suffix (str): Suffix to select weights that should be normalized.
            For WeightNorm, using 'weight_g' normalizes the scale variables.
        verbose (bool): Print out number of normalized and unnormalized parameters.
    """
    norm_params = []
    unnorm_params = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': 'unnormalized', 'params': unnorm_params}]

    if verbose:
        print('{} normalized parameters'.format(len(norm_params)))
        print('{} unnormalized parameters'.format(len(unnorm_params)))

    return param_groups


class WNConv2d(nn.Module):
    """
    Description::Weight-normalized 2d convolution.

    :param in_channels: Number of channels in the input.
    :param out_channels: Number of channels in the output.
    :param kernel_size: Side length of each convolutional kernel.
    :param padding: Padding to add on edges of input.
    :param bias: Use bias in the convolution operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super(WNConv2d, self).__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)
        return x


class BatchNormStats2d(nn.Module):
    """Compute BatchNorm2d normalization statistics: `mean` and `var`.
    Useful for keeping track of sum of log-determinant of Jacobians in flow models.
    Args:
        num_features (int): Number of features in the input (i.e., `C` in `(N, C, H, W)`).
        eps (float): Added to the denominator for numerical stability.
        decay (float): The value used for the running_mean and running_var computation.
            Different from conventional momentum, see `nn.BatchNorm2d` for more.
    """

    def __init__(self, num_features, eps=1e-5, decay=0.1):
        super(BatchNormStats2d, self).__init__()
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.decay = decay

    def forward(self, x, training):
        # Get mean and variance per channel
        if training:
            channels = x.transpose(0, 1).contiguous().view(x.size(1), -1)
            used_mean, used_var = channels.mean(-1), channels.var(-1)
            curr_mean, curr_var = used_mean, used_var

            # Update variables
            self.running_mean = self.running_mean - \
                self.decay * (self.running_mean - curr_mean)
            self.running_var = self.running_var - \
                self.decay * (self.running_var - curr_var)
        else:
            used_mean = self.running_mean
            used_var = self.running_var

        used_var += self.eps

        # Reshape to (N, C, H, W)
        used_mean = used_mean.view(1, x.size(1), 1, 1).expand_as(x)
        used_var = used_var.view(1, x.size(1), 1, 1).expand_as(x)

        return used_mean, used_var
