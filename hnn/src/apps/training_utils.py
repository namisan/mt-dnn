#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#
""" Utils for training and optimization
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils
logger=utils.get_logger()
import numpy as np
import torch
from bert.optimization import BertAdam

def zero_grad(model, optimizer_param):
  model.zero_grad()
  for n, p in optimizer_param:
    p.grad = None

def dump_parameter_names(model, path):
  with open(path, 'w', encoding='utf8') as fs:
    fs.write('{}\n'.format('\n'.join([n for n,p in model.named_parameters()])))

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
  """ Utility function for optimize_on_cpu and 16-bits training.
    Copy the parameters optimized on CPU/RAM back to the model on GPU
  """
  for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
    if name_opti != name_model:
      logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
      raise ValueError
    param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
  """ Utility function for optimize_on_cpu and 16-bits training.
    Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
  """
  is_nan = False
  for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
    if name_opti != name_model:
      logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
      raise ValueError
    if param_model.grad is not None:
      norm = param_model.grad.norm()
      if test_nan and (torch.isnan(norm) or torch.isinf(norm)):
        is_nan = True
      if param_opti.grad is None:
        param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
      param_opti.grad.data.copy_(param_model.grad.data)
    else:
      param_opti.grad = None
  return is_nan

def create_optimizer(model, args, num_train_steps=None, init_spec=None, no_decay=['bias', 'LayerNorm.weight']):
  # Prepare optimizer
  if args.fp16:
    dcnt = torch.cuda.device_count()
    if args.no_even_grad:
      param_optimizer = [(n, param.detach().clone().type(torch.cuda.FloatTensor).\
      requires_grad_()) for i,(n,param) in enumerate(model.named_parameters())]
    else:
      total_size = sum(np.prod(p.size()) for p in model.parameters())
      quota={i:0 for i in range(dcnt)}
      quota[0]=total_size//(dcnt*2)
      param_optimizer = []
      for i,(n, param) in enumerate(model.named_parameters()):
        ps = np.prod(param.size())
        index = list(sorted(quota.items(), key=lambda x: x[1]))[0][0]
        quota[index]+=ps
        cp = param.clone().type(torch.cuda.FloatTensor).detach().to('cuda:{}'.format(index)).requires_grad_()
        param_optimizer += [(n, cp)]
  elif args.optimize_on_cpu:
    param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
              for n, param in model.named_parameters()]
  else:
    param_optimizer = [(n,p) for n,p in model.named_parameters()]
  group0=dict(params=[],
      weight_decay_rate=args.weight_decay,
      names=[])
  group1=dict(params=[],
      weight_decay_rate=0.00,
      names=[])
  for (n,p) in param_optimizer:
    if not any(nd in n for nd in no_decay):
      group0['params'].append(p)
      group0['names'].append(n)
    else:
      group1['params'].append(p)
      group1['names'].append(n)

  optimizer_grouped_parameters = [group0, group1]
  t_total = num_train_steps
  optimizer=None

  if t_total:
    if args.local_rank != -1:
      t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
            lr=args.learning_rate,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
            v1=args.qhadam_v1,
            v2=args.qhadam_v2,
            lr_ends=args.lr_schedule_ends,
            e=args.epsilon,
            warmup=args.warmup_proportion if args.warmup_proportion<1 else args.warmup_proportion/t_total,
            t_total=t_total,
            schedule=args.lr_schedule,
            max_grad_norm=args.max_grad_norm,
            global_grad_norm=args.global_grad_norm,
            init_spec = init_spec,
            weight_decay_rate = args.weight_decay)
  return optimizer, param_optimizer, t_total
