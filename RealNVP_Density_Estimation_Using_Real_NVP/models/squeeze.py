#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :squeeze.py
@Description  :
@Date         :2022/02/21 16:59:24
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn.functional as F


def squeeze_2x2(x, invert=False, alt_order=False):
    """
    Description::For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
        reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.

    :param x: Input tensor of shape (B, C, H, W).
    :param invert: Whether to do a invert squeeze (unsqueeze).
    :param alt_order: Whether to use alternate ordering.

    Usage::
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    block_size = 2
    if alt_order:
        bs, c, h, w = x.size()

        if invert:
            if c % 4 != 0:
                raise ValueError(
                    f'Number of channels must be divisible by 4, got {c}.')
            c //= 4
        else:
            if h % 2 != 0:
                raise ValueError(f'Height must be divisible by 2, got {h}.')
            if w % 2 != 0:
                raise ValueError(f'Width must be divisible by 2, got {w}.')
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros(
            (4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                        + [c_idx * 4 + 1 for c_idx in range(c)]
                                        + [c_idx * 4 + 2 for c_idx in range(c)]
                                        + [c_idx * 4 + 3 for c_idx in range(c)])
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if invert:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            x = F.conv2d(x, perm_weight, stride=2)
    else:
        # x shape: [256, 3, 32, 32]
        bs, c, h, w = x.size()
        # out shape: [256, 32, 32, 3]
        x = x.permute(0, 2, 3, 1)

        if invert:
            if c % 4 != 0:
                raise ValueError(
                    f'Number of channels {c} is not divisible by 4')
            x = x.view(bs, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(bs, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(
                    f'Expected even spatial dims HxW, got {h}x{w}')
            # out shape: [256, 16, 2, 16, 2, 3]
            x = x.view(bs, h // 2, 2, w // 2, 2, c)
            # out shape: [256, 16, 16, 3, 2, 2]
            x = x.permute(0, 1, 3, 5, 2, 4)
            # out shape: [256, 16, 16, 12]
            x = x.contiguous().view(bs, h // 2, w // 2, c * 4)
        x = x.permute(0, 3, 1, 2)
    return x


def checkerboard_mask(height, width, reverse=False, dtype=torch.float32, device=None, requires_grad=False):
    """
    Description::Get a checkerboard mask, such that no two entries adjacent entries 
    have the same value. In non-reversed mask, top-left entry is 0.

    :param height: Number of rows in the mask.
    :param width: Number of columns in the mask.
    :param reverse: If True, reverse the mask (i.e., make top-left entry 1). Useful 
                    for alternating masks in RealNVP.
    :param dtype: Data type of the tensor.
    :param device: Device on which to construct the tensor.
    :param requires_grad: Whether the tensor requires gradient.

    :returns mask: Checkerboard mask of shape (1, 1, height, width).
    """

    checkerboard = [[((i % 2) + j) % 2 for j in range(width)]
                    for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype,
                        device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)
    return mask
