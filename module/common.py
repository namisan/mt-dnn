# Copyright (c) Microsoft. All rights reserved.
import torch
import math
from torch.nn.functional import tanh, relu, prelu, leaky_relu, sigmoid, elu, selu

def linear(x):
    return x

def swish(x):
    return x * sigmoid(x)

def gelu(x):
    """ref:https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L113
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def activation(func_a):
    """Activation function wrapper
    """
    try:
        f = eval(func_a)
    except:
        f = linear
    return f
