#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :main.py
@Description  :
@Date         :2022/02/21 16:42:45
@Author       :Arctic Little Pig
@version      :1.0
'''

import argparse
import os
import time

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from config_args import get_args
from load_data import get_data
from models.norm import get_param_groups
from models.real_nvp import RealNVP
from run_epoch import run_epoch


def get_param_numbers(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = get_args(argparse.ArgumentParser())

    train_loader, valid_loader, test_loader, predict_loader = get_data(args)

    model = RealNVP(num_scales=args.num_scales, in_channels=args.in_channels,
                    mid_channels=args.mid_channels, num_blocks=args.num_blocks)

    def load_saved_model(saved_model_name, model):
        checkpoint = torch.load(saved_model_name)
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint, model

    print(args.model_name)
    print("")

    if torch.cuda.device_count() > 1:
        print(f"---> Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()

    start_epoch = 1
    if args.resume:
        # Load checkpoint.
        print(f'---> resuming from checkpoint at {args.saved_model_name} ...')
        checkpoint, model = load_saved_model(args.saved_model_name, model)
        best_loss = checkpoint['loss']
        start_epoch = max(start_epoch, checkpoint['epoch'])

    if args.generate:
        model = load_saved_model(args.saved_model_name, model)

        imgs = run_epoch(args, model, None, None, 1,
                         "Generating", train=False, invert=True)
        images_concat = make_grid(imgs, nrow=int(
            args.num_samples ** 0.5), padding=2, pad_value=255)
        save_image(images_concat, os.path.join(args.sample_name, 'sample.png'))

        exit(0)

    param_groups = get_param_groups(
        model, args.weight_decay, norm_suffix='weight_g')

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, lr=args.lr)
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

    # print('---> trained model has {} parameters'.format(get_param_numbers(model)))

    # save all losses, epoch times, accuracy
    try:
        best_loss
    except NameError:
        best_loss = 0
    train_loss_all = []
    valid_loss_all = []
    test_loss_all = []
    epoch_time_all = []

    print("---> start training...")
    for epoch in range(start_epoch, args.epochs + start_epoch):
        print(
            f'======================== epoch: {epoch} ========================')

        mean_likelihood = 0.0
        num_minibatches = 0

        ################### Train ##################
        t_start = time.time()

        avg_loss_epoch = run_epoch(
            args, model, train_loader, optimizer, epoch, "Training", train=True)
        print(
            f'--> epoch{epoch} completed, average loss in training: {avg_loss_epoch}')
        train_loss_all.append(avg_loss_epoch)

        epoch_time_all.append(time.time() - t_start)

        ################### Valid ##################
        if valid_loader is not None:
            avg_loss_epoch = run_epoch(
                args, model, valid_loader, None, epoch, "Validating")
            print(f'--> average loss in validating: {avg_loss_epoch}')
            valid_loss_all.append(avg_loss_epoch)

        ################### Test ##################
        avg_loss_epoch = run_epoch(
            args, model, test_loader, None, epoch, "Testing")
        print(f'--> average loss in testing: {avg_loss_epoch}')
        test_loss_all.append(avg_loss_epoch)

        step_scheduler.step()

        if avg_loss_epoch < best_loss:
            print('---> saving...')
            state = {
                'state_dict': model.state_dict(),
                'loss': avg_loss_epoch,
                'epoch': epoch,
            }
            best_loss = avg_loss_epoch
            torch.save(state, os.path.join(
                args.model_name, 'model_' + str(epoch) + '.pt'))
