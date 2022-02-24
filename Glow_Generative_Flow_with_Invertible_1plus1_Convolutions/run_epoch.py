#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :run_epoch.py
@Description  :
@Date         :2022/02/22 20:41:24
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch
from tqdm import tqdm

from utils.common import AverageMeter, calc_z_shapes
from utils.loss import calc_loss


def run_epoch(args, model, data_loader, optimizer, epoch, desc, train=False, invert=False):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    if invert:
        with torch.no_grad():
            z_sample = []
            z_shapes = calc_z_shapes(
                3, args.img_size, args.num_flow, args.num_block)
            for z in z_shapes:
                z_new = torch.randn(args.num_sample, *z) * args.temp
                z_sample.append(z_new.cuda())
            x = model.reverse(z_sample).cpu().data
            return x

    loss_meter = AverageMeter()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (x, _) in pbar:
        # x shape: [bs, c, w, h]
        x = x.cuda()
        x = x * 255
        # print(x.shape, x)
        if args.num_bits < 8:
            x = torch.floor(x / 2 ** (8 - args.num_bits))
        x = x / args.num_bins - 0.5

        if train:
            if epoch == 1 and batch_idx == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model(
                        x + torch.rand_like(x) / args.num_bins)
                    continue
            else:
                log_p, logdet, _ = model(
                    x + torch.rand_like(x) / args.num_bins)
        else:
            with torch.no_grad():
                log_p, logdet, _ = model(
                    x + torch.rand_like(x) / args.num_bins)

        logdet = logdet.mean()
        loss, log_p, log_det = calc_loss(
            log_p, logdet, args.img_size, args.num_bins)

        loss_meter.update(loss.item(), x.size(0))

        if train:
            loss.backward()
            # Grad Accumulation
            if ((batch_idx + 1) % args.grad_ac_steps == 0):
                # warmup_lr = args.lr * \
                #     min(1, ((epoch - 1) * len(data_loader) +
                #         batch_idx * args.batch_size) / (50000 * 10))
                warmup_lr = args.lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()
                optimizer.zero_grad()
        pbar.set_description(
            f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}")
    return loss.item(), log_p.item(), log_det.item()
