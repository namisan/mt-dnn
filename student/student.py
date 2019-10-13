import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Student(nn.Module):
    def __init__(self, opt):
        super(Student, self).__init__()
        self.opt = opt
        self.eval_embed = None

    def forward(self, args):
        pass
