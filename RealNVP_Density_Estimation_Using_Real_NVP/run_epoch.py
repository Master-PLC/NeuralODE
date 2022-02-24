#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :run_epoch.py
@Description  :
@Date         :2022/02/21 16:42:55
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
from tqdm import tqdm

from utils.common import AverageMeter, bits_per_dim, clip_grad_norm
from utils.loss import RealNVPLoss


def run_epoch(args, model, data_loader, optimizer, epoch, desc, train=False, invert=False):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    if invert:
        with torch.no_grad():
            z = torch.randn((args.num_samples, 3, 32, 32), dtype=torch.float32)
            z = z.cuda()
            x, _ = model(z, invert=True)
            x = torch.sigmoid(x)
            return x

    criterion = RealNVPLoss()
    loss_meter = AverageMeter()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (x, _) in pbar:
        # x shape: [bs, c, w, h]
        x = x.cuda()
        # print(x.min(), x.max())

        if train:
            # out shape: [bs, 10]
            z, sldj = model(x)
        else:
            with torch.no_grad():
                z, sldj = model(x)
        loss = criterion(z, sldj)
        loss_meter.update(loss.item(), x.size(0))

        if train:
            loss.backward()
            # Grad Accumulation
            if ((batch_idx + 1) % args.grad_ac_steps == 0):
                # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                clip_grad_norm(optimizer, args.clip)
                optimizer.step()
                optimizer.zero_grad()
        pbar.set_postfix(loss=loss_meter.avg,
                         bpd=bits_per_dim(x, loss_meter.avg))
        pbar.update(x.size(0))
    return loss_meter.avg
