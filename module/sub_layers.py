# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class LayerNorm(nn.Module):
    #ref: https://github.com/pytorch/pytorch/issues/1959
    #   :https://arxiv.org/pdf/1607.06450.pdf
    def __init__(self, hidden_size, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(torch.ones(1,1,hidden_size)) # gain g
        self.beta = Parameter(torch.zeros(1,1,hidden_size)) # bias b
        self.eps = eps

    def forward(self, x):
        """
        Args:
            :param x: batch * len * input_size

        Returns:
            normalized x
        """
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x) + self.beta.expand_as(x)

class RnnEncoder(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='LSTM'):
        """
        RNN encoder wrapper.
        Note that the last hidden should be the first embedding for BERT/RoBERTa like model.
        """
        super(RnnEncoder, self).__init__()
        assert isinstance(rnn_type, str)
        rnn_type = rnn_type.upper()

        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        try:
            rnn_cls = getattr(nn, rnn_type)
        except:
            print('invalid RNN type: {}'.format(rnn_type))
            rnn_cls = getattr(nn, 'LSTM')
        # encoder = nn.ModuleList()
        # max_out layer class
        # 
        # for _ in range(nlayers):
            rnn = {}
            
        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
        self.do_max_out = self.ndirections > 1

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(),
                    weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        # last hidden state
        if self.ndirections == 1:
            last = output[:, -1]
        else:
            forward_ = output[:, -1, :self.num_hid]
            backward = output[:, 0, self.num_hid:]
            last = torch.cat((forward_, backward), dim=1)
        return output, last
