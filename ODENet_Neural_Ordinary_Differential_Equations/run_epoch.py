#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :run_epoch.py
@Description  :
@Date         :2022/02/12 15:55:35
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn as nn
from tqdm import tqdm


def run_epoch(args, model, data_loader, optimizer, epoch, desc, train=False):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    num_items = 0
    accuracy = 0.0
    run_losses = []
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data = data.cuda()
        target = target.cuda()
        # print(data.shape, target.shape)

        if train:
            # out shape: [bs, 10]
            output = model(data)
        else:
            with torch.no_grad():
                output = model(data)
        loss = criterion(output, target)

        if train:
            loss.backward()
            # Grad Accumulation
            if ((batch_idx + 1) % args.grad_ac_steps == 0):
                # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()

        run_losses += [loss.item()]
        num_items += data.shape[0]
        accuracy += torch.sum(torch.argmax(output,
                                           dim=1) == target).item()
    accuracy = accuracy * 100 / num_items

    return run_losses, accuracy
