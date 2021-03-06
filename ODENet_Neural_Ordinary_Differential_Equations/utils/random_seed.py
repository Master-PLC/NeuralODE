#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :random_seed.py
@Description  :
@Date         :2022/02/12 13:02:18
@Author       :Arctic Little Pig
@version      :1.0
'''

import random

import numpy as np
import torch

SEED = 20200913


def seed_init(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
