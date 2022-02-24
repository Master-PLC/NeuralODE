#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :trans_dataset.py
@Description  :
@Date         :2022/02/23 19:37:19
@Author       :Arctic Little Pig
@version      :1.0
'''

import numpy as np
from torch.utils.data import Dataset


class TransDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
