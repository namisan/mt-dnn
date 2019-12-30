#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#
""" Winograd Schema Challenge model for common sense reasoning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor

import csv
import os
import json

import random
import time
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn

from bert.modeling import *
from module import *

import utils
logger=utils.get_logger()
import pdb
from collections.abc import Mapping
from copy import copy

class HNNEncoder(NNModule):
  """HNN model
  """
  def __init__(self, config):
    super(HNNEncoder, self).__init__(config)
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.apply(self.init_weights)
    self.config = config

  def forward(self, input_ids, token_type_ids=None, input_mask=None, output_all_encoded_layers=True, position_ids = None, return_att = False):
    if input_mask is None:
      input_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    extended_input_mask = input_mask.unsqueeze(1).unsqueeze(2)

    embedding_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids, input_mask)
    encoded_layers = self.encoder(embedding_output,
                                extended_input_mask,
                                output_all_encoded_layers=output_all_encoded_layers, return_att=return_att)
    if return_att:
      encoded_layers, att_matrixs = encoded_layers
    if not output_all_encoded_layers:
      encoded_layers = encoded_layers[-1:]

    if return_att:
      return (encoded_layers, att_matrixs)
    return encoded_layers

class Cosine(torch.nn.Module):
  def __init__(self, config):
    super().__init__()

  def forward(self, src, tgt):
    src = src.float()
    tgt = tgt.float()
    return (torch.matmul(src, tgt.transpose(2,1))/(src.norm(p=2, dim=-1,keepdim=True)*tgt.norm(p=2, dim=-1, keepdim=True)+1e-9)).squeeze()

class BiLinearSim(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

  def forward(self, src, tgt):
    src_ = self.linear(src)
    output = torch.matmul(src_, tgt.transpose(2,1))
    return output

def binary_loss(x, labels, alpha=10, beta=0.5, gama=1):
  pos = ((x)*labels).sum(-1)
  neg = ((x + beta)*(1-labels)).sum(-1)
  
  loss = (-torch.nn.functional.logsigmoid(gama*pos)-torch.nn.functional.logsigmoid(-neg*gama))/2
  return loss

def binary_rank_loss(x, labels, alpha=10, beta=0.5, gama=1):
  logp = torch.nn.functional.logsigmoid(x)
  log1_p = -x + logp

  prob = torch.exp(logp)
  # suppose we only have two candidates
  pos_idx = labels.nonzero()[:,1].view(x.size(0), 1)
  neg_idx = 1 - pos_idx
  pos = torch.gather(prob, dim=-1, index=pos_idx.long()).squeeze(-1)
  neg = torch.gather(prob, dim=-1, index=neg_idx.long()).squeeze(-1)

  loss = -(labels*logp).sum(-1) + alpha*torch.max(torch.zeros_like(pos), neg-pos+beta)
  return loss

def rank_loss(x, labels, alpha=10, beta=0.5, gama=1):
  p = x
  logp = (torch.log(p)*labels).sum(-1)
  pos = (p*labels).sum(-1)
  neg = (p*(1-labels)).sum(-1)

  delta =  beta*(1-labels)+x

  loss = -(torch.nn.LogSoftmax(-1)(gama*delta)*labels).sum(-1)
  return loss

_Similarity={'cos': Cosine, 'bilinear':BiLinearSim}

_loss={'binary': binary_loss}

class SSMatcher(torch.nn.Module):
  """ Semantic similarity matcher
  """
  def __init__(self, config, alpha = 5, beta = 0.1, gama = 1, similarity='cos', \
      loss_type='binary', pooling='cap'):
    super().__init__()
    self.alpha = alpha
    self.beta = beta
    self.gama = gama
    self.config = config
    self.sim = similarity
    self.similarity = _Similarity[similarity](config)
    assert pooling in ['mean', 'cap', 'ftp'], 'Only cap, mean, ftp are supported pooling methods.'
    self.pooling = pooling
    if pooling=='cap':
      self.query = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.loss_fn = _loss[loss_type]

  def forward(self, bert, input_ids, labels=None, return_att=False):
    #if str(input_ids.device)=='cuda:0':
    #  pdb.set_trace()
    # expanded candidates
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(-1)

    input_ids = input_ids.view(-1, *input_ids.size()[2:])
    token_ids, mask_ids, type_ids, candidate_masks, pronoun_mask = [x.squeeze(1).contiguous() for x in input_ids.split(1, dim=1)]

    encoder_layers = bert(token_ids, type_ids, mask_ids, output_all_encoded_layers=True, return_att=return_att)
    if return_att:
      encoder_layers, att_matrixs = encoder_layers
    ctx_layer = encoder_layers[-1]

    pronoun_encoding = self.calc_pronoun_encoding(ctx_layer, pronoun_mask)

    # [bxc]x1xd
    if self.pooling == 'mean':
      pooling_fn = self.calc_candidate_encoding_mean
    elif self.pooling == 'ftp':
      pooling_fn = self.calc_candidate_encoding_ftp
    else:
      pooling_fn = self.calc_candidate_encoding_cap
    candidate_encoding,att_score = pooling_fn(ctx_layer, pronoun_encoding, candidate_masks)
    if return_att:
      att_matrixs.append(att_score)

    sim_score = self.similarity(candidate_encoding, pronoun_encoding).view(batch_size, -1)
    cands_id = (candidate_masks.sum(dim=-1)>0).to(sim_score).view(batch_size, -1)
    logits = sim_score + -10000*(1-cands_id)
    pred_probs = torch.sigmoid(logits)
    loss = torch.zeros(1).to(logits)
    if self.training:
      assert labels is not None
      # b x n
      labels = labels.view(batch_size, -1)
      x = logits

      # b x c x d
      cand_ebd = candidate_encoding.view(batch_size, -1, candidate_encoding.size(-1))
      # bx1

      loss = self.loss_fn(x, labels, self.alpha, self.beta, self.gama)
      loss = loss.mean()
      if torch.isnan(loss) or torch.isinf(loss):
        pdb.set_trace()

    if self.sim=='cos':
      logits = (logits+1)/2
    else:
      logits = pred_probs

    if return_att:
      return logits, loss, att_matrixs
    return (logits, loss)

  def calc_pronoun_encoding(self, context_layer, pronoun_mask):
    ctx = context_layer[:,0,:].unsqueeze(1)
    query = ctx
    att = torch.matmul(query, context_layer.transpose(2,1))/math.sqrt(query.size(-1))
    att_score = XSoftmax.apply(att, pronoun_mask.unsqueeze(1), -1)
    pronoun_ebd = torch.matmul(att_score, context_layer)
    return pronoun_ebd

  # wwm 75.4
  # CAP
  def calc_candidate_encoding_cap(self, context_layer, pronoun_encoding, candidate_masks):
    #bx1xd
    ctx = context_layer[:,0,:].unsqueeze(1)
    #if str(context_layer.device)=='cuda:1':
    #  pdb.set_trace()
    query = self.query(ctx)
    att = torch.matmul(query, context_layer.transpose(2,1))/math.sqrt(query.size(-1))
    att_score = XSoftmax.apply(att, candidate_masks.unsqueeze(1), -1)
    cand_ebd = torch.matmul(att_score, context_layer)
    return cand_ebd, att_score
  
  # Mean pooling
  def calc_candidate_encoding_mean(self, context_layer, pronoun_encoding, candidate_masks):
    #bx1xd
    ctx = context_layer[:,0,:].unsqueeze(1)
    query = torch.zeros_like(ctx)
    att = torch.matmul(query, context_layer.transpose(2,1))/math.sqrt(query.size(-1))
    att_score = XSoftmax.apply(att, candidate_masks.unsqueeze(1), -1)
    cand_ebd = torch.matmul(att_score, context_layer)
    return cand_ebd, att_score

  # FTP First token pooling
  def calc_candidate_encoding_ftp(self, context_layer, pronoun_encoding, candidate_masks):
    #bx1xd
    ctx = context_layer[:,0,:].unsqueeze(1)
    idx = torch.arange(candidate_masks.size(-1),0,-1).unsqueeze(0)\
        .expand(candidate_masks.size()).to(candidate_masks)
    idx = idx*candidate_masks
    _, first_idx = torch.max(idx, dim=1, keepdim=True)
    first_idx = first_idx.unsqueeze(-1).expand([context_layer.size(0), 1, context_layer.size(-1)])
    cand_ebd = torch.gather(context_layer, dim=1, index=first_idx)
    return cand_ebd, None

def lm_loss(loglogits, labels, alpha, beta, gama):
    selected = (loglogits*labels).sum(-1)
    unselected = (loglogits*(1-labels)).sum(-1)
    delta = beta*(1-labels) + torch.exp(loglogits)
    delta = -(torch.nn.LogSoftmax(-1)(gama*delta)*labels).sum(-1)
    loss = -selected + alpha*delta
    return loss

class LMMatcher(torch.nn.Module):
  """ Langumage model matcher
  """
  def __init__(self, bert, config, alpha = 5, beta = 0.1, gama = 1):
    super().__init__()
    self.alpha = alpha
    self.beta = beta
    self.gama = gama
    self.lm_predictions = BertLMPredictionHead(config, bert.embeddings.word_embeddings.weight)

  def forward(self, bert, input_ids, labels=None):
    # expanded candidates
    is_expanded = False
    batch_size = input_ids.size(0)
    if input_ids.size(1)>2:
      is_expanded = True
    input_ids = input_ids.view(-1, *input_ids.size()[2:])
    input_ids,attention_mask,token_type_ids,input_labels,_ = [x.squeeze(1).contiguous() for x in input_ids.split(1, dim=1)]

    if is_expanded:
      # bc x 1
      valid_instances = input_labels.sum(-1)>0
      valid_index = valid_instances.nonzero()
      valid_index_ex = valid_index.expand(valid_index.size(0), input_ids.size(1))
      input_ids = torch.gather(input_ids, dim=0, index=valid_index_ex)
      input_labels = torch.gather(input_labels, dim=0, index=valid_index_ex)
      attention_mask = torch.gather(attention_mask, dim=0, index=valid_index_ex)
      token_type_ids = torch.gather(token_type_ids, dim=0, index=valid_index_ex)

    encoder_layers = bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
    ebd_weight = bert.embeddings.word_embeddings.weight
    ctx_layer = encoder_layers[-1]
    #bc x s x V
    mask_logits = self.lm_predictions(ctx_layer, ebd_weight).float()
    log_logits = nn.LogSoftmax(-1)(mask_logits)
    
    #bc x s
    label_logits = torch.gather(log_logits, dim=-1, index=input_labels.unsqueeze(-1).long()).squeeze(-1)
    label_mask = (input_labels != 0)
    #bc
    pred_logits = label_logits.masked_fill_(input_labels==0, 0).sum(-1)/label_mask.to(label_logits).sum(-1)
    
    if is_expanded:
      logits_ex = torch.zeros_like(valid_instances.to(pred_logits)).fill_(-1000.0)
      #if str(input_ids.device)=='cuda:0':
      #  pdb.set_trace()
      logits_ex.scatter_(dim=0, index=valid_index.squeeze(-1), src=pred_logits)
      pred_logits = logits_ex

    #b x c
    pred_logits = pred_logits.view(batch_size, -1)

    loss = torch.zeros(1).to(pred_logits)
    if self.training:
      assert labels is not None
      # all candidates are 2 or zeros
      # b x 1
      labels = labels.view(batch_size, -1)
      loss = lm_loss(pred_logits, labels, self.alpha, self.beta, self.gama)
      loss = loss.mean()
      if torch.isnan(loss) or torch.isinf(loss):
        pdb.set_trace()

    return (pred_logits.exp(), loss)

class HNNClassifer(NNModule):
  """ HNN model
  """
  def __init__(self, config, drop_out=None, alpha = [10,10], beta = [0.5,0.5], gama = [1,1], similarity='cos', loss_type='binary', pooling='cap'):
    super().__init__(config)
    self.bert = HNNEncoder(config)
    self.sm_matcher = SSMatcher(config, alpha[0], beta[0], gama[0], similarity, loss_type, pooling)

    lm_idx = 1 if len(alpha)>1 else 0
    self.lm_matcher = LMMatcher(self.bert, config, alpha[lm_idx], beta[lm_idx], gama[lm_idx])
    self.theta = torch.nn.Parameter(torch.tensor([1,1], dtype=torch.float))
    self.config = config
    self.loss_fn = _loss[loss_type]
    self.alpha = alpha
    self.beta = beta
    self.gama = gama
    self.apply(self.init_weights)

  def forward(self, input_ids, tids, labels=None, return_att=False, group_tasks=True):
    #if str(input_ids.device)=='cuda:0':
    #  pdb.set_trace()
    # expanded candidates
    group_tasks = bool(group_tasks[0]) if isinstance(group_tasks, torch.Tensor) else group_tasks
    sm_ids = (tids == 0).nonzero()

    lm_ids = (tids == 1).nonzero()

    def extract_tasks(task):
      if group_tasks:
        task = task[:,1].view(input_ids.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        task = task.expand((task.size(0), task.size(1), input_ids.size(-2), input_ids.size(-1))).long()
        task_inputs = torch.gather(input_ids, dim=1, index=task).contiguous()
        return task_inputs, labels
      else:
        task = task[:,0].view(-1, input_ids.size(1))
        input_idx = task.unsqueeze(-1).unsqueeze(-1).\
            expand((task.size(0), task.size(1), input_ids.size(-2), input_ids.size(-1))).long()
        task_inputs = torch.gather(input_ids, dim=0, index=input_idx).contiguous()
        #if str(input_ids.device)=='cuda:0':
        #  pdb.set_trace()
        task_labels = torch.gather(labels, dim=0, index=task).contiguous()
        return task_inputs, task_labels

    loss = torch.zeros(1).to(input_ids.device).float()
    sm_logits = None
    if len(sm_ids)>0:
      sm_inputs, sm_labels=extract_tasks(sm_ids)
      sm_logits, sm_loss = self.sm_matcher(self.bert, sm_inputs, sm_labels)
      loss += sm_loss

    lm_logits = None
    if len(lm_ids)>0:
      lm_inputs, lm_labels=extract_tasks(lm_ids)
      lm_logits, lm_loss = self.lm_matcher(self.bert, lm_inputs, lm_labels)
      loss += lm_loss

    en_logits = None
    #if sm_logits is not None and lm_logits is not None and group_tasks:
    #if sm_logits is not None and lm_logits is not None and group_tasks:
    if group_tasks or sm_logits is None or lm_logits is None:
      if sm_logits is None:
        en_logits = lm_logits
      elif lm_logits is None:
        en_logits = sm_logits
      else:
        en_logits = (sm_logits + lm_logits)/2
      if self.training:
        en_loss = rank_loss(en_logits, labels, self.alpha[-1], self.beta[-1], self.gama[-1])
        loss += en_loss.mean()
    if self.training:
      return None,None,None,loss
    else:
      return sm_logits, lm_logits, en_logits, loss
