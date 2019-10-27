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
from bert import BertConfig
from .tf_utils import *

from utils import get_logger
logger=get_logger()

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
