#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :cnf_test.py
@Description  :
@Date         :2022/02/18 20:28:48
@Author       :Arctic Little Pig
@version      :1.0
'''

import time
from pickletools import optimize

import numpy as np
import torch
import torch.nn as nn


class ODE_FUNC(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ODE_FUNC, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fx = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.ft = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, t):
        self.x = x
        self.t = t
        self.out = self.fx(x) + self.ft(t)
        return self.out

    def get_trace(self):
        result = 0
        for i in range(self.out_dim):
            # 计算trace。注意的是，这里是取出第i个变量下第i位的数据并求和得到的结果。
            dfdx = torch.autograd.grad(self.out[:, i].sum(), 
                                       self.x, 
                                       retain_graph=True, 
                                       create_graph=True)[0]
            result += dfdx.contiguous()[:, i].contiguous()
        return result


class CNF(object):
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.ode_func = ODE_FUNC(in_dim, hidden_dim, out_dim)

    def get_simulation(self, x0, t0, step, T):
        x = x0  # shape: [120, 1]
        t = t0  # shape: [120, 1]
        d_trace = 0

        for i in range(T):
            f = self.ode_func(x, t)  # shape: [120, 1]
            d_trace += self.ode_func.get_trace()

            x = x + f * step
            t = t + step
        return x, d_trace


if __name__ == "__main__":
    in_dim = 1
    hidden_dim = 6
    out_dim = 1

    batch_size = 120
    step = 0.1
    T = 10

    cnf = CNF(in_dim, hidden_dim, out_dim)
    optimizer = torch.optim.SGD(
        cnf.ode_func.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(10000000):
        optimizer.zero_grad()

        x = np.random.normal(loc=1, scale=0.2, size=batch_size)
        x = torch.from_numpy(x).float()
        x = torch.unsqueeze(x, 1)  # shape: [120, 1]
        x.requires_grad_(True)

        t = torch.zeros(batch_size, 1, requires_grad=True)

        x_final, d_trace = cnf.get_simulation(x, t, step, T)

        loss_e = 0.5 * torch.mean(x_final ** 2)
        loss = torch.mean((loss_e - d_trace) ** 2)

        loss.backward()
        if epoch % 1000 == 0:
            print(loss)
            print(f'mean := {torch.mean(x_final)}')

        optimizer.step()
