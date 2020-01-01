# Author: penhe@microsoft.com
# Date: 05/30/2019
#
""" Data parallel module
"""

from collections import OrderedDict
import numpy as np
import torch
from torch.cuda.comm import broadcast_coalesced
from torch.cuda.comm import reduce_add_coalesced
from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs,gather
import torch.cuda.comm as comm
import pdb
from bert.optimization import BertAdam


def replicate(network, devices):

  devices = tuple(devices)
  num_replicas = len(devices)

  params = list(network.parameters())
  param_indices = {param: idx for idx, param in enumerate(params)}
  param_copies = broadcast_coalesced(params, devices)

  buffers = list(network._all_buffers())
  buffer_indices = {buf: idx for idx, buf in enumerate(buffers)}
  buffer_copies = broadcast_coalesced(buffers, devices)

  modules = list(network.modules())
  module_copies = [[] for device in devices]
  module_indices = {}

  for i, module in enumerate(modules):
    module_indices[module] = i
    for j in range(num_replicas):
      replica = module.__new__(type(module))
      replica.__dict__ = module.__dict__.copy()
      replica._parameters = replica._parameters.copy()
      replica._buffers = replica._buffers.copy()
      replica._modules = replica._modules.copy()
      module_copies[j].append(replica)

  for i, module in enumerate(modules):
    for key, child in module._modules.items():
      if child is None:
        for j in range(num_replicas):
          replica = module_copies[j][i]
          replica._modules[key] = None
      else:
        module_idx = module_indices[child]
        for j in range(num_replicas):
          replica = module_copies[j][i]
          replica._modules[key] = module_copies[j][module_idx]
    for key, param in module._parameters.items():
      if param is None:
        for j in range(num_replicas):
          replica = module_copies[j][i]
          replica._parameters[key] = None
      else:
        param_idx = param_indices[param]
        for j in range(num_replicas):
          replica = module_copies[j][i]
          replica._parameters[key] = param_copies[j][param_idx]
          replica._parameters[key].requires_grad = param.requires_grad
    for key, buf in module._buffers.items():
      if buf is None:
        for j in range(num_replicas):
          replica = module_copies[j][i]
          replica._buffers[key] = None
      else:
        buffer_idx = buffer_indices[buf]
        for j in range(num_replicas):
          replica = module_copies[j][i]
          replica._buffers[key] = buffer_copies[j][buffer_idx]

  return [module_copies[j][0] for j in range(num_replicas)]

class XDataParallel(torch.nn.Module):
  def __init__(self, module):
    super().__init__()
    self.device_ids = [i for i in range(torch.cuda.device_count())]
    module = module.cuda(self.device_ids[0])
    self.replicas = replicate(module, self.device_ids)
    self.output_device = self.device_ids[0]
    self.dim = 0
    self.module = module

  def forward(self, *inputs, **kwargs):
    #if not self.device_ids:
    #  return self.module(*inputs, **kwargs)
    inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
    #if len(self.device_ids) == 1:
    #  return self.module(*inputs[0], **kwargs[0])
    #replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(self.replicas[:len(inputs)], inputs, kwargs)
    return self.gather(outputs, self.output_device)

  def state_dict(self, destination=None, prefix='', keep_vars=False):
    sd = self.replicas[0].state_dict(destination, prefix, keep_vars)
    return sd

  def eval(self):
    for m in self.replicas:
      m.eval()
    return self

  def train(self, mode=True):
    for m in self.replicas:
      m.train(mode)
    return self

  def zero_grad(self):
    for m in self.replicas:
      for p in m.parameters():
        p.grad = None

  def scatter(self, inputs, kwargs, device_ids):
    return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

  def parallel_apply(self, replicas, inputs, kwargs):
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

  def gather(self, outputs, output_device):
    return gather(outputs, output_device, dim=self.dim)

class XParallelOptimizer():
  def __init__(self, model, optimizer_fn, grad_clip_norm=1.0):
    self.replicas = [model]
    if hasattr(model, 'replicas'):
      self.replicas = model.replicas
    dcnt = torch.cuda.device_count()
    total_size = sum([np.prod(p.size()) for p in self.replicas[0].parameters()])
    quota = {i:0 for i in range(dcnt)}
    #quota[0] = total_size//dcnt
    param_groups = {i: [] for i in range(dcnt)}
    self.named_parameters=[]
    for i,(n, param) in enumerate(self.replicas[0].named_parameters()):
      ps = np.prod(param.size())
      index = list(sorted(quota.items(), key=lambda x: x[1]))[0][0]
      quota[index] += ps
      if param.dtype==torch.half:
        cp = param.clone().type(torch.cuda.FloatTensor).detach().to('cuda:{}'.format(index)).requires_grad_()
      else:
        cp = dict(self.replicas[index].named_parameters())[n]
      name = n[len('module.'):] if n.startswith('module.') else n
      param_groups[index].append((name, cp))
      self.named_parameters.append((name, cp))
    self.param_groups = param_groups
    self.sub_optimizers = [DeviceOptimizer(self.replicas, p, i, optimizer_fn(p, max_grad_norm=0)) for i,p in self.param_groups.items()]
    self.grad_clip_norm = grad_clip_norm

  def parameters(self):
    return OrderedDict(self.named_parameters)

  def step(self, grad_scale=1):
    def bk(g):
      return g.backward()
    l2norm_square = parallel_apply([bk for _ in self.sub_optimizers], self.sub_optimizers, devices=[g.device for g in self.sub_optimizers])
    l2norm = sum(l2norm_square)**0.5
    
    if str(l2norm) in ['inf', 'nan']:
      return False

    if grad_scale != 1:
      l2norm *= grad_scale
    coef = self.grad_clip_norm/(l2norm+1e-6)
    
    if coef<1:
      grad_scale = grad_scale*coef
    if grad_scale != 1:
      for n,p in self.named_parameters:
        if p.grad is not None:
          p.grad.mul_(grad_scale)
    
    def st(g):
      return g.step(l2norm)
    parallel_apply([st for _ in self.sub_optimizers], self.sub_optimizers, devices=[g.device for g in self.sub_optimizers])
    
    return True

  def zero_grad(self):
    for m in self.replicas:
      for p in m.parameters():
        p.grad = None
    for g in self.sub_optimizers:
      g.zero_grad()

class DeviceOptimizer():
  def __init__(self, replicas, param_group, device, optimizer):
    self.param_group = param_group
    self.device = device
    self.optimizer = optimizer
    self.replicas = replicas
    self.named_params = [dict(m.named_parameters()) for m in replicas]

  def backward(self):
    group_params = [[(n,m[n]) for n,p in self.param_group if m[n].grad is not None] for m in self.named_params]
    grad_params = [g for g in group_params if len(g)>0]
    assert all([len(g)==len(grad_params[0]) for g in grad_params]), [len(g) for g in grad_params]
    grad = [[p.grad for n,p in g] for g in grad_params]
    reduced_grad = reduce_add_coalesced(grad, self.device)
    grads = dict([(n,g) for ((n,p),g) in zip(grad_params[0], reduced_grad)])
    l2norm = 0
    for n,p in self.param_group:
      if n in grads:
        p.grad = grads[n].float() if grads[n].dtype==torch.half else grads[n]
        l2norm += p.grad.norm().item()**2
      else:
        assert p.grad is None, n
    return l2norm

  def step(self, l2norm):
    self.optimizer.step()
    group_params = [(i, [(n,m[n]) for n,p in self.param_group]) for i,m in enumerate(self.named_params)]
    group_params = sorted(group_params, key=lambda x:x[0] if x[0]!=self.device else -1)
    params = dict(self.param_group)
    for n,p in group_params[0][1]:
      if p.data.dtype == torch.half:
        p.data.copy_(params[n].data)
      else:
        p.data = params[n].data
    param_list = [[p for n,p in g] for i,g in group_params]
    device_list =[i for i,g in group_params]
    outputs = broadcast_coalesced(param_list[0], device_list)
    for o,p in zip(outputs, param_list):
      for x,y in zip(o, p):
        y.data.copy_(x.data)

  def zero_grad(self):
    for n,p in self.param_group:
      p.grad = None
    self.optimizer.zero_grad()

def optimizer_factory(args, training_steps=None, init_spec=None, no_decay=['bias', 'LayerNorm.weight']):
  def optimizer_fn(param_group, max_grad_norm=None):
    group0 = dict(params=[],
      weight_decay_rate=args.weight_decay,
      names=[])
    group1 = dict(params=[],
      weight_decay_rate=0.00,
      names=[])
    for (n,p) in param_group:
      if not any(nd in n for nd in no_decay):
        group0['params'].append(p)
        group0['names'].append(n)
      else:
        group1['params'].append(p)
        group1['names'].append(n)

    optimizer_grouped_parameters = [group0, group1]

    optimizer = BertAdam(optimizer_grouped_parameters,
            lr=args.learning_rate,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
            v1=args.qhadam_v1,
            v2=args.qhadam_v2,
            lr_ends=args.lr_schedule_ends,
            warmup=args.warmup_proportion if args.warmup_proportion<1 else args.warmup_proportion/training_steps,
            t_total=training_steps,
            schedule=args.lr_schedule,
            max_grad_norm = args.max_grad_norm if max_grad_norm is None else max_grad_norm,
            global_grad_norm = args.global_grad_norm,
            init_spec = init_spec,
            weight_decay_rate = args.weight_decay)
    return optimizer

  return optimizer_fn
