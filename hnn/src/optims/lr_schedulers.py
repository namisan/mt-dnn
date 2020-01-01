#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#
""" Learning rate schedulers
"""

import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_

def warmup_cosine(step, total, warmup=0.002, ends = 0):
    x = step/total
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_constant(step, total, warmup=0.002, ends = 0):
    x = step/total
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(step, total, warmup=0.002, ends = 0):
    x = step/total
    if x < warmup:
        return x/warmup
    return (1-ends)*(1.0 - x) + ends

def warmup_linear_cosine(step, total, warmup=0.002, ends = 0):
    x = step/total
    if x < warmup:
        return x/warmup
    return (1-ends)*max(0.5*(1+math.cos(math.pi*(x-warmup)/(1-warmup))), 0) + ends

def warmup_cyclic_linear_cosine(step, total, warmup=0.002, ends = 0):
    x = step/total
    if x < warmup:
        return x/warmup
    total = total - int(total*warmup)
    step = step - int(total*warmup)
    n_epoch = 4
    period = total//n_epoch
    k = step//period
    s = 1-k/n_epoch + 1/(2*n_epoch)*(math.pow(-1, k)*math.cos(math.pi*step/period)-1)
    return (1-ends)*max(s, 0) + ends

def warmup_linear_shift(step, total, warmup=0.002, ends = 0):
    x = step/total
    if x < warmup:
        return x/warmup
    return 1.0 - x + warmup

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_linear_cosine':warmup_linear_cosine,
    'warmup_cyclic_linear_cosine':warmup_cyclic_linear_cosine,
    'warmup_linear_shift':warmup_linear_shift,
}
