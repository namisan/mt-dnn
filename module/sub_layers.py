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

import copy
from pytorch_pretrained_bert.modeling import BertEmbeddings, BertLayerNorm, BertConfig
from module.similarity import SelfAttnWrapper
from module.dropout_wrapper import DropoutWrapper

class MaxOut(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [batch, seq_len, hidden_size * 2]
        size = x.shape
        assert size[-1] % 2 == 0
        return max_output

class SanLayer(nn.Module):
    def __init__(self, num_hid, bidirect, dropout, rnn_type):
        super().__init__()

        assert isinstance(rnn_type, str)
        rnn_type = rnn_type.upper()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = getattr(nn, rnn_type)
        self._rnn = rnn_cls(num_hid, num_hid, 1,
                bidirectional=bidirect,
                dropout=dropout,
                batch_first=True)
        if bidirect:
            self._max_out = MaxOut()
        self._layer_norm = BertLayerNorm(num_hid, eps=1e-12)
        self.rnn_type = rnn_type
        self.num_hid = num_hid
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(),
                    weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()
    
    def forward(self, x, attention_mask):
        # x: [batch, sequence, in_dim]
        self._rnn.flatten_parameters()

        batch = x.size(0)
        hidden0 = self.init_hidden(batch)

        tmp_output = self._rnn(x, hidden0)[0]
        if self.ndirections > 1:
            size = tmp_output.shape
            tmp_output = tmp_output.view(size[0], size[1], self.num_hid, 2).max(-1)[0]
        output = self._layer_norm(x + tmp_output)
        return output

class SanEncoder(nn.Module):
    def __init__(self, num_hid, nlayers, bidirect, dropout, rnn_type='LSTM'):
        super().__init__()
        layer = SanLayer(num_hid, bidirect, dropout, rnn_type)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(nlayers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class SanPooler(nn.Module):
    def __init__(self, hidden_size, dropout_p, enable_vbp):
        super().__init__()
        my_dropout = DropoutWrapper(dropout_p, enable_vbp)
        self.self_att = SelfAttnWrapper(hidden_size, dropout=my_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        """
        
        Arguments:
            hidden_states {FloatTensor} -- shape (batch, seq_len, hidden_size)
            attention_mask {ByteTensor} -- 1 indicates padded token
        """
        first_token_tensor =  self.self_att(hidden_states, attention_mask)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SanNetwork(nn.Module):
    def __init__(self, config: BertConfig, opt):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = SanEncoder(config.hidden_size, config.num_hidden_layers, True, 
                                  config.hidden_dropout_prob)
        self.pooler = SanPooler(config.hidden_size, config.hidden_dropout_prob, opt['vb_dropout'])
        self.config = config

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """[summary]
        
        Arguments:
            input_ids {LongTensor} -- shape [batch_size, seq_len]
        
        Keyword Arguments:
            token_type_ids {LongTensor} -- shape [batch_size, seq_len]
            attention_mask {LongTensor} -- 0 indicates padding tokens
        
        Returns: Tuple of (sequence_output, pooled_output)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output, attention_mask, output_all_encoded_layers=False)[-1]
        pooled_output = self.pooler(sequence_output, attention_mask == 0)
        return sequence_output, pooled_output
