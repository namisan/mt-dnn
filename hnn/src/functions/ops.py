# Author: penhe@microsoft.com
# Date: 04/25/2019
#
""" Functions optimized for training
"""

__all__=['XSoftmax', 'StableDropout', 'MaskedLayerNorm', 'BertLayerNorm']
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import pdb
from packaging import version

if version.Version(torch.__version__) >= version.Version('1.0.0'):
  from torch import _softmax_backward_data as _softmax_backward_data
else:
  from torch import softmax_backward_data as _softmax_backward_data

from torch.nn import LayerNorm as BertLayerNorm

class XSoftmax(torch.autograd.Function):
  @staticmethod
  def forward(self, input, mask, dim):
    self.dim = dim
    if version.Version(torch.__version__) >= version.Version('1.2.0a'):
      rmask = (1-mask).bool()
    else:
      rmask = (1-mask).byte()

    output = input.masked_fill(rmask, float('-inf'))
    output = torch.softmax(output, self.dim)
    output.masked_fill_(rmask, 0)
    self.save_for_backward(output)
    return output

  @staticmethod
  def backward(self, grad_output):
    output, = self.saved_tensors
    inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
    return inputGrad, None, None

class XDropout(torch.autograd.Function):
  @staticmethod
  def forward(self, input, dropout):
    self.dropout = dropout
    self.scale=1/(1-dropout)
    if self.dropout>0:
      if version.Version(torch.__version__) >= version.Version('1.2.0a'):
        mask=(1-torch.empty_like(input).bernoulli_(1-self.dropout)).bool()
      else:
        mask=(1-torch.empty_like(input).bernoulli_(1-self.dropout)).byte()

      self.save_for_backward(mask)
      return input.masked_fill(mask, 0)*self.scale
    else:
      return input

  @staticmethod
  def backward(self, grad_output):
    if self.dropout > 0:
      mask, = self.saved_tensors
      return grad_output.masked_fill(mask, 0)*self.scale, None
    else:
      return grad_output, None

class StableDropout(nn.Module):
  def __init__(self, drop_prob):
    super(StableDropout, self).__init__()
    self.drop_prob = drop_prob
    self.dropout=torch.nn.Dropout(drop_prob)
  def forward(self, x):
    if self.training and self.drop_prob>0:
      return XDropout.apply(x, self.drop_prob)
    return x

def MaskedLayerNorm(layerNorm, input, mask = None):
  output = layerNorm(input)
  if mask is None:
    return output

  if mask.dim()==4:
    mask=mask.squeeze(1).squeeze(1)
  mask = mask.unsqueeze(2).to(input.dtype)
  return output*mask
