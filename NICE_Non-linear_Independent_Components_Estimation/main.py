#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :main.py
@Description  :
@Date         :2022/02/19 18:28:41
@Author       :Arctic Little Pig
@version      :1.0
'''

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config_args import get_args
from load_data import get_data
from models.nice import NICE
from run_epoch import run_epoch


def get_param_numbers(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = get_args(argparse.ArgumentParser())

    train_loader, valid_loader, test_loader, predict_loader = get_data(args)

    model = NICE(data_dim=784, hidden_dim=args.hidden_dim,
                 num_coupling_layers=args.cp_layers, num_net_layers=args.layers)

    def load_saved_model(saved_model_name, model):
        checkpoint = torch.load(saved_model_name)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    print(args.model_name)
    print("")

    if torch.cuda.device_count() > 1:
        print(f"---> Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()

    if args.generate:
        model = load_saved_model(args.saved_model_name, model)
        if test_loader is not None:
            data_loader = test_loader
        else:
            data_loader = valid_loader

        img = run_epoch(args, model, data_loader, None, 1,
                        "Generating", train=False, invert=True)
        pil_img = Image.fromarray(np.uint8(img[1]))
        pil_img.show()
        exit(0)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise Exception('Unknown optimizer')

    if args.scheduler_type == 'plateau':
        step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5)
    elif args.scheduler_type == 'step':
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        step_scheduler = None

    print('---> trained model has {} parameters'.format(get_param_numbers(model)))

    # save all losses, epoch times, accuracy
    train_likelihood = []
    valid_likelihood = []
    test_likelihood = []
    epoch_time_all = []

    print("---> start training...")
    for epoch in range(1, args.epochs + 1):
        print(
            f'======================== epoch: {epoch} ========================')

        mean_likelihood = 0.0
        num_minibatches = 0

        ################### Train ##################
        t_start = time.time()

        mean_likelihood = run_epoch(
            args, model, train_loader, optimizer, epoch, "Training", train=True)
        print(
            f'--> epoch{epoch} completed, log Likelihood in training: {mean_likelihood}')
        train_likelihood.append(mean_likelihood)

        epoch_time_all.append(time.time() - t_start)

        ################### Valid ##################
        if valid_loader is not None:
            mean_likelihood = run_epoch(
                args, model, valid_loader, None, epoch, "Validating")
            print(
                f'--> log Likelihood in validating: {mean_likelihood}')
            valid_likelihood.append(mean_likelihood)

        ################### Test ##################
        mean_likelihood = run_epoch(
            args, model, test_loader, None, epoch, "Testing")
        print(f'--> log Likelihood in testing: {mean_likelihood}')
        test_likelihood.append(mean_likelihood)

        step_scheduler.step()

        if epoch % args.save_every == 0:
            torch.save({'state_dict': model.state_dict()}, os.path.join(
                args.model_name, 'model_' + str(epoch) + '.pt'))
