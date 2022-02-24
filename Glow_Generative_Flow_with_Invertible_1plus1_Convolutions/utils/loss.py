#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :loss.py
@Description  :
@Date         :2022/02/23 21:04:48
@Author       :Arctic Little Pig
@version      :1.0
'''


from math import log


def calc_loss(log_p, logdet, image_size, num_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(num_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )
