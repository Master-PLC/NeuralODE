#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :run_epoch.py
@Description  :
@Date         :2022/02/19 20:32:45
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
import torch.nn as nn
from tqdm import tqdm


def run_epoch(args, model, data_loader, optimizer, epoch, desc, train=False, invert=False):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    if invert:
        with torch.no_grad():
            z = model.prior.sample([10, 784])
            x = model(z, invert=True)
            x = x.view(-1, 28, 28).cpu()
            return x

    mean_likelihood = 0.0
    num_minibatches = 0

    for batch_idx, (x, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        x = x.view(-1, 784) + torch.rand(784) / 256
        x = x.cuda()
        x = torch.clamp(x, 0, 1)
        # print(x.shape)

        if train:
            # out shape: [bs, 10]
            z, likelihood = model(x)
        else:
            with torch.no_grad():
                z, likelihood = model(x)
        loss = -torch.mean(likelihood)

        if train:
            loss.backward()
            # Grad Accumulation
            if ((batch_idx + 1) % args.grad_ac_steps == 0):
                # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()

        mean_likelihood -= loss
        num_minibatches += 1

    mean_likelihood /= num_minibatches
    return mean_likelihood
