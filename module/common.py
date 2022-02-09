# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import math
from torch.nn.init import (
    uniform,
    normal,
    eye,
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    orthogonal,
)


def linear(x):
    return x


def swish(x):
    return x * sigmoid(x)


def bertgelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gptgelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


# default gelue
gelu = bertgelu


def activation(func_a):
    """Activation function wrapper"""
    try:
        f = eval("nn.{}".format(func_a))
    except:
        f = linear
    return f


def init_wrapper(init="xavier_uniform"):
    return eval(init)
