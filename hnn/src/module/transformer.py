#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#
""" Transformer
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

import copy
import numpy as np
import torch
import torch.nn as nn
import pdb
from collections.abc import Mapping

from bert.modeling import *
from .pooling import *
from utils import get_logger
logger=get_logger()

class InputEncoder(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """
  def __init__(self, config):
    super(InputEncoder, self).__init__()
    self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
    # any TensorFlow checkpoint file
    #self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
    self.dropout = StableDropout(config.hidden_dropout_prob)

  def forward(self, input_ids, token_type_ids=None, position_ids=None, mask = None):
    seq_length = input_ids.size(1)
    if position_ids is None:
      position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = words_embeddings + position_embeddings + token_type_embeddings
    embeddings = self.dropout(embeddings)
    if mask is not None:
      if mask.dim()==4:
        mask=mask.squeeze(1).squeeze(1)
      mask = mask.unsqueeze(2).to(embeddings)
      embeddings = embeddings * mask
    #embeddings = MaskedLayerNorm(self.LayerNorm, embeddings, mask)
    return embeddings

  def freeze(self):
    self.dropout.drop_prob = 0

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            #print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

#V1
class SelfAttention(nn.Module):
  def __init__(self, config, head_num = None):
    super(SelfAttention, self).__init__()
    self.num_attention_heads = config.num_attention_heads if head_num is None else head_num
    if config.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
    #self.query = WeightDrop(nn.Linear(config.hidden_size, self.all_head_size, bias=False), ['weight'], config.attention_probs_dropout_prob)
    self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
    self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

    self.dropout = StableDropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask):
    mixed_query_layer = self.query(hidden_states)
    #mixed_query_layer = self.dropout(self.query(hidden_states))
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    #try:
    #  from apex.softmax import MaskedSoftmax
      #attention_mask = self.dropout(attention_mask.repeat(1, self.num_attention_heads, 1, 1).half()).to(torch.int32)
      #attention_mask = self.dropout(attention_mask.half()).to(torch.int32)
    #  attention_probs = MaskedSoftmax(dim=-1)(attention_scores, attention_mask)
    #  if torch.isnan(attention_probs.norm()):
    #    logger.error("Hit invalid attention score. NaN")
    #    raise ValueError("Hit invalid attention score")
    #except Exception as ex:
    attention_mask = attention_mask.to(dtype=hidden_states.dtype) # fp16 compatibility
    attention_scores = attention_scores*attention_mask + (1.0-attention_mask)*(-10000.0)
    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return context_layer

class SelfAttentionV2(nn.Module):
  def __init__(self, config):
    super(SelfAttentionV2, self).__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
    self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
    self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
    
    self.att_weight = torch.nn.Parameter(torch.zeros([self.num_attention_heads, 1, self.attention_head_size]))
    torch.nn.init.xavier_uniform_(self.att_weight.data)

    self.dropout = StableDropout(config.attention_probs_dropout_prob)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)
    mask = attention_mask.permute(0,1,3,2).to(value_layer.dtype)
    query_scores = torch.matmul(query_layer, self.att_weight.permute(0,2,1))*mask
    key_scores = torch.matmul(key_layer, self.att_weight.permute(0,2,1))*mask
    length = torch.sum(mask, dim = -2, keepdim=True)
    mean_value = torch.sum(value_layer*mask, dim = -2, keepdim=True)
    query_values = torch.matmul(query_scores, mean_value)
    key_values = torch.matmul(key_scores.permute(0,1,3,2), value_layer)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    context_layer = (query_values + key_values)/length
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return context_layer

class AttentionProject(nn.Module):
  def __init__(self, config):
    super(AttentionProject, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.config = config

  def forward(self, hidden_states, input_tensor, mask=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = ACT2FN[self.config.att_act](hidden_states)
    return hidden_states

class AttentionBlock(nn.Module):
  def __init__(self, config, head_num = None):
    super(AttentionBlock, self).__init__()
    self.self = SelfAttention(config, head_num = head_num)
    #self.self = SelfAttentionV2(config)
    self.output = AttentionProject(config)
    self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)

  def forward(self, input_tensor, attention_mask):
    self_output = self.self(self.LayerNorm(input_tensor), attention_mask)
    attention_output = self.output(self_output, input_tensor, attention_mask)
    return attention_output

class ExpandProject(nn.Module):
  def __init__(self, config):
    super(ExpandProject, self).__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
    self.intermediate_act_fn = ACT2FN[config.hidden_act] \
      if isinstance(config.hidden_act, str) else config.hidden_act

  def forward(self, hidden_states):
    hidden_states = self.dense(self.LayerNorm(hidden_states))
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states

class BackProject(nn.Module):
  def __init__(self, config):
    super(BackProject, self).__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

  def forward(self, hidden_states, input_tensor, mask=None):
    hidden_states = self.dense(hidden_states)
    return hidden_states

class TransformerBlock(nn.Module):
  def __init__(self, config, head_num = None):
    super(TransformerBlock, self).__init__()
    self.attention = AttentionBlock(config, head_num = head_num)
    self.intermediate = ExpandProject(config)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.output = BackProject(config)

  def forward(self, hidden_states, attention_mask):
    attention_output = self.attention(hidden_states, attention_mask)
    expand_input = hidden_states + self.dropout(attention_output)
    intermediate_output = self.intermediate(expand_input)
    layer_output = self.output(intermediate_output, attention_output, attention_mask)
    return self.dropout(layer_output) + expand_input

  def freeze(self):
    self.output.dropout.drop_prob = 0
    self.attention.output.dropout.drop_prob = 0
    self.attention.self.dropout.drop_prob = 0

class TransformerEncoder(nn.Module):
  def __init__(self, config):
    super(TransformerEncoder, self).__init__()
    #layer = TransformerBlock(config)
    #self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
    head_num = config.num_attention_heads if isinstance(config.num_attention_heads, list) else [config.num_attention_heads]
    self.layer = nn.ModuleList([TransformerBlock(config, head_num=head_num[i%len(head_num)]) for i in range(config.num_hidden_layers)])

  def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
    all_encoder_layers = []
    for layer_module in self.layer:
      hidden_states = layer_module(hidden_states, attention_mask)
      if output_all_encoded_layers:
        all_encoder_layers.append(hidden_states)
    if not output_all_encoded_layers:
      all_encoder_layers.append(hidden_states)
    return all_encoder_layers

  def freeze(self, layers):
    for i in layers:
      self.layer[i].freeze()

class LMMaskPredictionHeadV1(nn.Module):
    def __init__(self, config, max_seq):
        super(LMMaskPredictionHeadV1, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
        self.classifer = nn.Linear(config.hidden_size, max_seq)

    def forward(self, hidden_states):
        # b x d
        ctx_states = self.LayerNorm(hidden_states[:,0,:])
        ctx_states = self.dense(ctx_states)
        ctx_states = self.transform_act_fn(ctx_states)

        # b x max_len
        logits = self.classifer(ctx_states)
        # truncate to sequence length
        logits = logits[:,:hidden_states.size(-2)].contiguous()
        return logits

class LMMaskPredictionHeadV2(nn.Module):
    def __init__(self, config, max_seq):
        super(LMMaskPredictionHeadV2, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
        self.classifer = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        # b x d
        ctx_states = hidden_states[:,0,:]
        seq_states = self.LayerNorm(ctx_states.unsqueeze(-2) + hidden_states)
        seq_states = self.dense(seq_states)
        seq_states = self.transform_act_fn(seq_states)

        # b x max_len
        logits = self.classifer(seq_states).squeeze(-1)
        # truncate to sequence length
        # logits = logits[:,:hidden_states.size(-2)].contiguous()
        return logits

class LMMaskPredictionHeadV3(nn.Module):
    def __init__(self, config, max_seq):
        super(LMMaskPredictionHeadV3, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size*2, 1e-7)
        self.classifer = nn.Linear(config.hidden_size*2, 1)

    def forward(self, hidden_states):
        # b x d
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        ctx_states = hidden_states[:,0,:]
        seq_states = self.LayerNorm(torch.cat([ctx_states.unsqueeze(-2).expand(hidden_states.size()), hidden_states], dim=-1))

        # b x max_len
        logits = self.classifer(seq_states).squeeze(-1)
        # truncate to sequence length
        # logits = logits[:,:hidden_states.size(-2)].contiguous()
        return logits

class LMMaskPredictionHeadV4(nn.Module):
    def __init__(self, config, max_seq):
        super(LMMaskPredictionHeadV4, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
        self.classifer = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        # b x d
        hidden_states = self.dense(self.LayerNorm(hidden_states))
        seq_states = self.transform_act_fn(hidden_states)

        # b x max_len
        logits = self.classifer(seq_states).squeeze(-1)
        # truncate to sequence length
        # logits = logits[:,:hidden_states.size(-2)].contiguous()
        return logits

class NNModule(nn.Module):
  """ An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
  """

  def __init__(self, config, *inputs, **kwargs):
    super().__init__()
    if not isinstance(config, BertConfig):
      raise ValueError(
        "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
        "To create a model from a Google pretrained model use "
        "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
          self.__class__.__name__, self.__class__.__name__
        ))
    self.config = config

  def init_weights(self, module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if hasattr(module, 'weight_raw'):
          truncated_normal_init(module.weight_raw.data, mean=0, stdv=self.config.initializer_range)
      else:
          truncated_normal_init(module.weight.data, mean=0, stdv=self.config.initializer_range)

      #torch.nn.init.xavier_uniform_(module.weight.data)
    #elif isinstance(module, BertLayerNorm):
      # in tf, layer norm always inited with 0, 1 as beta and gamma
      #module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
      #truncated_normal_init(module.beta.data, mean=0, stdv=self.config.initializer_range)
      #module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
      #truncated_normal_init(module.gamma.data, mean=0, stdv=self.config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def partial_reset_weights(self, module, ratio = 0.1):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if hasattr(module, 'weight_raw'):
          weight = module.weight_raw
      else:
          weight = module.weight
      temp = torch.zeros_like(weight.data)
      mask = torch.zeros_like(weight.data).bernoulli_(ratio)
      truncated_normal_init(temp, mean=0, stdv=self.config.initializer_range)
      weight.data.copy_((mask*temp + weight.data*(1-mask)).detach())
      #torch.nn.init.xavier_uniform_(module.weight.data)
    #elif isinstance(module, BertLayerNorm):
      # in tf, layer norm always inited with 0, 1 as beta and gamma
      #module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
      #truncated_normal_init(module.beta.data, mean=0, stdv=self.config.initializer_range)
      #module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
      #truncated_normal_init(module.gamma.data, mean=0, stdv=self.config.initializer_range)
    #if isinstance(module, nn.Linear) and module.bias is not None:
    #  module.bias.data.zero_()

  @classmethod
  def load_model(cls, model_path, bert_config, init_spec, *inputs, **kwargs):
    """
    Instantiate a NNModule from a pre-trained model file.
    Download and cache the pre-trained model file if needed.
    
    Params:
      pretrained_model_name: either:
        - a str with the name of a pre-trained model to load selected in the list of:
          . `bert-base-uncased`
          . `bert-large-uncased`
          . `bert-base-cased`
          . `bert-base-multilingual`
          . `bert-base-chinese`
        - a path or url to a pretrained model archive containing:
          . `bert_config.json` a configuration file for the model
          . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
      *inputs, **kwargs: additional input for the specific Bert class
        (ex: num_labels for BertForSequenceClassification)
    """
    # Load config

    config = BertConfig.from_json_file(bert_config)
    logger.info("Model config {}".format(config))
    # Instantiate model.
    model = cls(config, *inputs, **kwargs)
    if model_path is None:
      return model
    logger.info("loading prtrained local model file {}".format(model_path))
    state_dict = torch.load(model_path, map_location='cpu')

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    for k in list(state_dict.keys()):
      if 'LayerNorm.gamma' in k:
        nk = k.replace('LayerNorm.gamma', 'LayerNorm.weight')
        state_dict[nk]=state_dict[k]
        del state_dict[k]
      if 'LayerNorm.beta' in k:
        nk = k.replace('LayerNorm.beta', 'LayerNorm.bias')
        state_dict[nk]=state_dict[k]
        del state_dict[k]

    ignore_init = []
    if init_spec:
      remap_dict = type(state_dict)()
      for var in init_spec:
        mapping = init_spec[var].mapping
        name = init_spec[var].name
        if name.startswith('module.'):
          name = name[len('module.'):]
        if (not init_spec[var].use_pretrain):
          ignore_init += [name]
        elif mapping:
          if mapping.startswith('module.'):
            mapping = mapping[len('module.'):]
          if mapping in state_dict:
            remap_dict[name] = state_dict[mapping]
      logger.info('Variables not using pretraining: {}'.format(ignore_init))
      for ig in ignore_init:
        if ig in state_dict:
          del state_dict[ig]
      for key in state_dict:
        if key not in remap_dict:
          remap_dict[key]=state_dict[key]
      state_dict = remap_dict

    if metadata is not None:
      state_dict._metadata = metadata

    def load(module, prefix=''):
      local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
      module._load_from_state_dict(
        state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
      for name, child in module._modules.items():
        if child is not None:
          load(child, prefix + name + '.')
    load(model)
    if len(missing_keys) > 0:
      logger.warn("Weights of {} not initialized from pretrained model: {}".format(
        model.__class__.__name__, '\n  '.join(missing_keys)))
    if len(unexpected_keys) > 0:
      logger.warn("Weights from pretrained model not used in {}: {}".format(
        model.__class__.__name__, '\n  '.join(unexpected_keys)))
    return model
