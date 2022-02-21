#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :main.py
@Description  :
@Date         :2022/02/12 12:28:57
@Author       :Arctic Little Pig
@version      :1.0
'''

import argparse
import os
import time

import numpy as np

from config_args import get_args
from load_data import get_data
from models.adjoint import NeuralODE
from models.resnet import *
from run_epoch import run_epoch


def get_param_numbers(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = get_args(argparse.ArgumentParser())

    print(f"---> use ode training: {args.use_ode}")

    train_loader, valid_loader, test_loader, predict_loader = get_data(args)

    if args.use_ode:
        func = ConvBlockOde(64)
        feat = NeuralODE(func, tol=args.tol, solver=args.solver)
    else:
        feat = nn.Sequential(*[ResBlock(64, 64) for _ in range(args.layers)])
    model = ContinuousResNet(feat, channels=args.num_channel)

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
    train_loss_all = []
    valid_loss_all = []
    test_loss_all = []
    epoch_time_all = []
    accuracy_all = []

    print("---> start training...")
    for epoch in range(1, args.epochs + 1):
        print(
            f'======================== epoch: {epoch} ========================')

        ################### Train ##################
        t_start = time.time()

        train_loss_epoch, _ = run_epoch(
            args, model, train_loader, optimizer, epoch, "Training", train=True)
        print(f'--> train loss: {np.mean(train_loss_epoch):.4f}')
        train_loss_all.append(train_loss_epoch)

        epoch_time_all.append(time.time() - t_start)

        ################### Valid ##################
        if valid_loader is not None:
            valid_loss_epoch, _ = run_epoch(
                args, model, valid_loader, None, epoch, "Validating")
            print(f'--> valid loss: {np.mean(valid_loss_epoch):.4f}')
            valid_loss_all.append(valid_loss_epoch)

        ################### Test ##################
        test_loss_epoch, accuracy = run_epoch(
            args, model, test_loader, None, epoch, "Testing")
        print(f'--> test loss: {np.mean(test_loss_epoch):.4f}')
        print(f"accuracy: {np.round(accuracy, 3)}%")
        test_loss_all.append(test_loss_epoch)
        accuracy_all.append(np.round(accuracy, 3))

        step_scheduler.step()

        if epoch % args.save_every == 0:
            torch.save({'state_dict': model.state_dict()}, os.path.join(
                args.model_name, 'model_' + str(epoch) + '.pt'))
        if epoch % args.log_every == 0:
            torch.save({'accuracy': accuracy_all,
                        'train_loss': train_loss_all,
                        'epoch_time': epoch_time_all}, os.path.join(args.model_name, 'log.pkl'))
