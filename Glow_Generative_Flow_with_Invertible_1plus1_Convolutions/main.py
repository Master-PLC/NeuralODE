#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :main.py
@Description  :
@Date         :2022/02/22 20:41:31
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
from models.glow import Glow
from run_epoch import run_epoch


def get_param_numbers(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_saved_model(saved_model_name, model):
    start_epoch = 1

    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['loss']
    start_epoch = max(start_epoch, checkpoint['epoch'])
    return model, optimizer, best_loss, start_epoch


if __name__ == "__main__":
    args = get_args(argparse.ArgumentParser())

    train_loader, valid_loader, test_loader, predict_loader = get_data(args)

    model = Glow(args.in_channels, args.num_flow, args.num_block,
                 affine=args.affine, conv_lu=not args.no_lu)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise Exception('Unknown optimizer')

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
        model, optimizer, best_loss, start_epoch = load_saved_model(
            args.saved_model_name, model)

    if args.generate:
        model, _, _, _ = load_saved_model(args.saved_model_name, model)

        imgs = run_epoch(args, model, None, None, 1,
                         "Generating", train=False, invert=True)
        save_image(imgs, os.path.join(args.sample_name, 'sample.png'),
                   normalize=True, nrow=10, range=(-0.5, 0.5))

        exit(0)

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

        ################### Train ##################
        t_start = time.time()

        loss, log_p, log_det = run_epoch(
            args, model, train_loader, optimizer, epoch, "Training", train=True)
        print(
            f'--> epoch{epoch} completed, loss in training: {loss}, logP in training: {log_p}.')
        train_loss_all.append(loss)

        epoch_time_all.append(time.time() - t_start)

        ################### Valid ##################
        if valid_loader is not None:
            loss, log_p, log_det = run_epoch(
                args, model, valid_loader, None, epoch, "Validating")
            print(
                f'--> loss in validating: {loss}, logP in validating: {log_p}.')
            valid_loss_all.append(loss)

        ################### Test ##################
        loss, log_p, log_det = run_epoch(
            args, model, test_loader, None, epoch, "Testing")
        print(f'--> loss in testing: {loss}, logP in testing: {log_p}.')
        test_loss_all.append(loss)

        step_scheduler.step()

        if loss < best_loss:
            print('---> saving...')
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'epoch': epoch,
            }
            best_loss = loss
            torch.save(state, os.path.join(
                args.model_name, 'model_' + str(epoch) + '.pt'))
