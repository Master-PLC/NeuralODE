#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :config_args.py
@Description  :
@Date         :2022/02/22 20:41:41
@Author       :Arctic Little Pig
@version      :1.0
'''

import os
import shutil
from xmlrpc.client import Boolean

from utils.random_seed import seed_init


def get_args(parser, eval=False):
    parser.add_argument('--dataroot', type=str, default='../datasets/')
    parser.add_argument('--dataset', type=str, choices=[
                        'mnist', 'cifar10', 'ag_news', 'celeba-hq'], default='celeba-hq')
    parser.add_argument('--train_ratio', type=float,
                        default=0.7, help='ratio of train dataset.')
    parser.add_argument('--test_ratio', type=float,
                        default=0.1, help='ratio of test dataset.')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--samples_dir', type=str, default='samples')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=89112)
    parser.add_argument('--gpu_id', type=str, default='0')

    # Optimization
    parser.add_argument('--optim', type=str,
                        choices=['adam', 'sgd', 'rmsprop'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate of optimizer")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--grad_ac_steps', type=int, default=1,
                        help="gradient accumulation steps, grad_ac_steps supply batch_size equals to real batch_size")
    parser.add_argument('--scheduler_step', type=int, default=40)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--int_loss', type=float, default=0.0)
    parser.add_argument('--aux_loss', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str,
                        choices=['bce', 'mixed', 'class_ce', 'soft_margin'], default='bce')
    parser.add_argument('--scheduler_type', type=str,
                        choices=['plateau', 'step'], default='step')
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--warmup_scheduler',
                        action='store_true', help='use warmup scheduler or step scheduler')
    parser.add_argument('--clip', type=float, default=100,
                        help='clip the gradient')

    # Model
    parser.add_argument('--num_flow', type=int, default=32,
                        help="number of flows in each block")
    parser.add_argument('--num_block', type=int,
                        default=4, help="number of blocks")
    parser.add_argument('--no_lu', default=False, action="store_true",
                        help="use plain convolution instead of LU decomposed version")
    parser.add_argument('--affine', default=False, action="store_true",
                        help="use affine coupling instead of additive")
    parser.add_argument("--num_bits", type=int,
                        default=5, help="number of bits")
    parser.add_argument("--temp", default=0.7, type=float,
                        help="temperature of sampling")
    parser.add_argument('--dropout', type=float, default=0.1)

    # Testing Models
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--num_samples', type=int,
                        default=20, help="number of samples")
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--saved_model_name', type=str,
                        default='results\mnist.4cp_layer.6layer.bsz_256.adam0.001.ep75\model_15.pt')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()

    if args.train_ratio + args.test_ratio > 1:
        raise ValueError(
            f"Ratio of train data and test data should be smaller than 1. Which now are {args.train_ratio} and {args.test_ratio}.")

    args.num_bins = 2 ** args.num_bits

    model_name = args.dataset
    if args.dataset == 'cifar10':
        args.in_channels = 3
        args.img_size = 32
    elif args.dataset == 'mnist':
        args.in_channels = 1
        args.img_size = 28
    elif args.dataset == 'celeba-hq':
        args.in_channels = 3
        args.img_size = 64
    else:
        print('dataset not included')
        exit()

    model_name += f'.{args.num_flow}flow'
    model_name += f'.{args.num_block}block'
    model_name += f'.{args.num_bits}bits'
    model_name += f'.temp{args.temp}'
    model_name += f'.bsz_{int(args.batch_size * args.grad_ac_steps)}'
    model_name += f'.{args.optim}{args.lr}'  # .split('.')[1]
    # model_name += f'.clip{args.clip}'  # .split('.')[1]
    model_name += f'.ep{args.epochs}'

    if args.int_loss != 0.0:
        model_name += f".int_loss{str(args.int_loss).split('.')[1]}"

    if args.aux_loss != 0.0:
        model_name += f".aux_loss{str(args.aux_loss).replace('.', '')}"

    if args.name != '':
        model_name += f'.{args.name}'

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    model_name = os.path.join(args.results_dir, model_name)

    args.model_name = model_name
    args.sample_name = os.path.join(model_name, args.samples_dir)

    if args.inference:
        args.epochs = 1

    if os.path.exists(args.model_name) and (not args.overwrite) and (not 'test' in args.name) and (not eval) and (not args.inference) and (not args.resume) and (not args.predict) and (not args.generate):
        print(args.model_name)
        overwrite_status = input('Already Exists. Overwrite?: ')
        if overwrite_status == 'rm':
            shutil.rmtree(args.model_name)
        elif not 'y' in overwrite_status:
            exit(0)
    elif not os.path.exists(args.model_name):
        os.makedirs(args.model_name)

    if not os.path.exists(args.sample_name):
        os.makedirs(args.sample_name)

    seed_init(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    return args
