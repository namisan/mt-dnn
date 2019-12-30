# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

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

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 v1=1, v2=1,
                 lr_ends = 0,
                 max_grad_norm=1.0, global_grad_norm=True, init_spec=None):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        self.defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        lr_ends = lr_ends,
                        max_grad_norm=max_grad_norm, global_grad_norm=global_grad_norm,
                        v1=v1, v2=v2)
        self.init_spec = init_spec
        self.global_step = 0
        super(BertAdam, self).__init__(params, self.defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, bs_scale = 1, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        def grad_frozen(n,p,g):
            state=self.state[p]
            t_total = g['t_total']
            spec = None
            if self.init_spec:
                spec = self.init_spec[n] if n in self.init_spec else None
            if spec is not None:
                fz = spec.freeze_ratio * t_total
                return self.global_step < fz
            return p.grad is None
        loss = None
        self.global_step += 1
        if closure is not None:
            loss = closure()
        params = [p for g in self.param_groups for (n,p) in zip(g['names'],g['params']) if g['lr']>0 and not grad_frozen(n, p, g)]
        if self.defaults['max_grad_norm'] > 0 and self.defaults['global_grad_norm']:
            gn = clip_grad_norm_(params, self.defaults['max_grad_norm'])
        for group in self.param_groups:
            for p,n in zip(group['params'], group['names']):
                if grad_frozen(n, p, group):
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)


                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0 and not group['global_grad_norm']:
                    gn = clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                qhg = 0
                if group['v1']<1:
                    qhg = (1/group['v1']-1)*grad
                qhg2 = 0
                if group['v2']<1:
                    qhg2 = (1/group['v2']-1)*grad*grad
                c = group['v1']/math.sqrt(group['v2'])
                update = (qhg + next_m) / ((qhg2 + next_v).sqrt() + group['e'])
                if c != 1:
                    update *= c;

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] > 0:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'], group['t_total'], group['warmup'], group['lr_ends'])
                else:
                    lr_scheduled = group['lr']
                state['step'] += 1
                if group['warmup']==0:
                  bias_correction = math.sqrt(1-math.pow(beta2, state['step']))/\
                    (1-math.pow(beta1, state['step']))
                  lr_scheduled *= bias_correction
                update_with_lr = (bs_scale*lr_scheduled) * update
                p.data.add_(-update_with_lr)

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
