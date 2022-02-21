#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :load_data.py
@Description  :
@Date         :2022/02/12 13:47:32
@Author       :Arctic Little Pig
@version      :1.0
'''

import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


def get_data(args):
    dataset = args.dataset
    data_root = args.dataroot
    batch_size = args.batch_size

    workers = args.workers

    if args.test_batch_size == -1:
        args.test_batch_size = batch_size

    valid_dataset = None
    valid_loader = None
    test_dataset = None
    test_loader = None
    predict_dataset = None
    predict_loader = None
    drop_last = True

    if dataset == 'mnist':
        mnist_root = os.path.join(data_root, 'mnist')
        img_std = 0.3081
        img_mean = 0.1307

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((img_mean,), (img_std,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((img_mean,), (img_std,))
        ])

        train_dataset = datasets.MNIST(
            root=mnist_root, train=True, download=True, transform=transform_train)

        valid_dataset = datasets.MNIST(
            root=mnist_root, train=True, download=True, transform=transform_test)

        test_dataset = datasets.MNIST(
            root=mnist_root, train=False, download=True, transform=transform_test)

    elif dataset == 'cifar10':
        cifar10_root = os.path.join(data_root, 'cifar10')

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(
            root=cifar10_root, train=True, download=True, transform=transform_train)
        
        test_dataset = datasets.CIFAR10(
            root=cifar10_root, train=False, download=True, transform=transform_test)

    else:
        print('no dataset available')
        exit(0)

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=workers, drop_last=drop_last)
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=workers)
    if predict_dataset is not None:
        predict_loader = DataLoader(
            predict_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=workers)

    return train_loader, valid_loader, test_loader, predict_loader
