#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#
"""
Pooling functions
"""

from bert.modeling import *
from torch import nn

class PoolConfig(object):
    """Configuration class to store the configuration of `attention pool layer`.
    """
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 att_act="none",
                 att_proj=True,
                 hidden_act="gelu",
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 shrink_size=None,
                 project_act="tanh",
                 project_dropout_prob=0,
                 project_size=None
                 ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.project_size = project_size
        self.project_dropout_prob = project_dropout_prob
        self.shrink_size = shrink_size
        self.project_act = project_act
        self.att_act = att_act
        self.att_proj = True

        self.context_layernorm = False
        self.context_drop = 0

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = cls()
        for key, value in json_object.items():
            config.__dict__[key] = value
        config.shrink_size = config.hidden_size if \
            (config.shrink_size is None or config.shrink_size<0) else config.shrink_size
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class ContextPooler(nn.Module):
    def __init__(self, config):
        super(ContextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.with_ln = config.context_layernorm if hasattr(config, 'context_layernorm') else False
        self.context_drop = config.context_drop if hasattr(config, 'context_drop') else 0
        if self.with_ln:
            self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
        self.dropout = StableDropout(self.context_drop)
        self.config = config

    def forward(self, hidden_states, mask = None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0]
        if self.with_ln:
          context_token = self.LayerNorm(context_token)
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN['gelu'](pooled_output)
        return pooled_output

    def output_dim(self):
        return self.config.hidden_size

class AttentionBlock(nn.Module):
    def __init__(self, config):
        super(AttentionBlock, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attention_project = nn.Linear(config.hidden_size, config.hidden_size)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_dropout = StableDropout(config.attention_probs_dropout_prob)

        self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
        self.output_dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_tensors, key_tensors, value_tensors, attention_mask, query_mask=None):
        mixed_query_layer = self.query(query_tensors)
        mixed_key_layer = self.key(key_tensors)
        mixed_value_layer = self.value(value_tensors)
        #mixed_value_layer = value_tensors

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        try:
            from apex.softmax import MaskedSoftmax
            #attention_mask = attention_mask.repeat(1, self.num_attention_heads, 1, 1).float()
            #attention_mask = self.attention_dropout(attention_mask.half()).to(torch.int32)
            attention_probs = MaskedSoftmax(dim=-1)(attention_scores, attention_mask.to(torch.int32))
        except Exception as ex:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.attention_project is not None:
            att_output = self.attention_project(context_layer)
            output = ACT2FN[self.config.att_act](self.output_dropout(att_output))
            output = MaskedLayerNorm(self.LayerNorm, output + query_tensors, query_mask)
        else:
            att_output = context_layer
            output = ACT2FN[self.config.att_act](self.output_dropout(att_output))
        return output

class AttentivePooler(nn.Module):
    def __init__(self, config):
        super(AttentivePooler, self).__init__()
        self.pooler = AttentionBlock(config)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        if config.intermediate_size > 0:
            self.expand = nn.Linear(config.hidden_size, config.intermediate_size)
            intermediate_size = config.intermediate_size
            if config.shrink_size > 0:
                self.shrink = nn.Linear(config.intermediate_size, config.shrink_size)
                intermediate_size = config.shrink_size
            if config.hidden_size==intermediate_size:
                self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
        else:
            intermediate_size = config.hidden_size

        self.dropout = StableDropout(config.hidden_dropout_prob)

        if config.project_size > 0:
            self.project = nn.Linear(intermediate_size, config.project_size)
            self.project_dropout = StableDropout(config.hidden_dropout_prob if config.project_dropout_prob is None else config.project_dropout_prob)
            self.project_act_fn = ACT2FN[config.project_act]

        self.config = config

    def forward(self, hidden_states, type_ids, input_mask):
        with torch.no_grad():
            attention_mask = input_mask.to(hidden_states.device)
            #type_ids = type_ids.to(hidden_states.device)
            #seq_len = attention_mask.sum(-1, keepdim=True)
            #type_a_len = seq_len - (type_ids*attention_mask).sum(-1, keepdim=True)
            #context_pos = torch.zeros_like(seq_len)
            # mask out special token
            #sep_idx = torch.cat((context_pos, type_a_len - 1, seq_len - 1), dim = -1)
            #sep_idx = torch.cat((type_a_len - 1, seq_len - 1), dim = -1)
            #attention_mask.scatter_(-1, sep_idx, 0)
            #type_a = (1-type_ids)*attention_mask
            #type_b = type_ids*attention_mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.device)
        pooled_output = self.pooler(hidden_states[:, 0].unsqueeze(1), hidden_states, hidden_states, attention_mask).squeeze(1)
        if self.config.intermediate_size > 0:
            output = self.expand(pooled_output)
            output = self.intermediate_act_fn(output)
            if self.config.shrink_size > 0:
                output = self.shrink(output)
            if pooled_output.size(-1)==output.size(-1):
                output = MaskedLayerNorm(self.LayerNorm, self.dropout(output) + pooled_output)
        else:
            output = pooled_output
        if self.config.project_size > 0:
            output = self.project_act_fn(self.project(self.project_dropout(output)))
        return output

    def output_dim(self):
        if self.config.project_size > 0:
            return self.config.project_size
        elif self.config.shrink_size > 0:
            return self.config.shrink_size
        elif self.config.intermediate_size > 0:
            return self.config.intermediate_size
        else:
            return self.config.hidden_size

class DualAttentionBlock(nn.Module):
  def __init__(self, config):
    super(DualAttentionBlock, self).__init__()
    if config.hidden_size % config.num_attention_heads != 0:
      raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    if config.att_proj:
      self.attention_project = nn.Linear(config.hidden_size, config.hidden_size)
    else:
      self.attention_project = None

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key_a = nn.Linear(config.hidden_size, self.all_head_size)
    self.value_a = nn.Linear(config.hidden_size, self.all_head_size)

    self.key_b = nn.Linear(config.hidden_size, self.all_head_size)
    self.value_b = nn.Linear(config.hidden_size, self.all_head_size)

    self.attention_dropout = StableDropout(config.attention_probs_dropout_prob)

    self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
    self.output_dropout = StableDropout(config.hidden_dropout_prob)
    self.config = config

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, query_tensors, key_tensors, value_tensors, attention_mask, type_a_mask=None, type_b_mask=None, query_mask=None):
    mixed_query_layer = self.query(query_tensors)
    type_a_mask = type_a_mask.to(query_tensors).unsqueeze(-1)
    type_b_mask = type_b_mask.to(query_tensors).unsqueeze(-1)
    mixed_key_layer = self.key_a(key_tensors)*type_a_mask + self.key_b(key_tensors)*type_b_mask
    mixed_value_layer = self.value_a(value_tensors)*type_a_mask + self.value_b(value_tensors)*type_b_mask
    #mixed_value_layer = value_tensors

    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    try:
      from apex.softmax import MaskedSoftmax
      #attention_mask = attention_mask.repeat(1, self.num_attention_heads, 1, 1).float()
      #attention_mask = self.attention_dropout(attention_mask.half()).to(torch.int32)
      attention_probs = MaskedSoftmax(dim=-1)(attention_scores, attention_mask.to(torch.int32))
    except Exception as ex:
      attention_mask = attention_mask.to(dtype=hidden_states.dtype) # fp16 compatibility
      attention_mask = (1.0 - attention_mask) * -10000.0
      attention_scores = attention_scores + attention_mask
      # Normalize the attention scores to probabilities.
      attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    if self.attention_project is not None:
      att_output = self.attention_project(context_layer)
    else:
      att_output = context_layer
    output = ACT2FN[self.config.att_act](self.output_dropout(att_output))
    #output = MaskedLayerNorm(self.LayerNorm, output + query_tensors, query_mask)
    return output

class DualAttentivePooler(nn.Module):
  def __init__(self, config):
    super(DualAttentivePooler, self).__init__()
    self.attention = DualAttentionBlock(config)
    self.intermediate_act_fn = ACT2FN[config.hidden_act] \
      if isinstance(config.hidden_act, str) else config.hidden_act
    if config.intermediate_size > 0:
      self.expand = nn.Linear(config.hidden_size, config.intermediate_size)
      intermediate_size = config.intermediate_size
      if config.shrink_size > 0:
        self.shrink = nn.Linear(config.intermediate_size, config.shrink_size)
        intermediate_size = config.shrink_size
      if config.hidden_size==intermediate_size:
        self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)
    else:
      intermediate_size = config.hidden_size

    self.dropout = StableDropout(config.hidden_dropout_prob)

    if config.project_size > 0:
      self.project = nn.Linear(intermediate_size, config.project_size)
      self.project_dropout = StableDropout(config.hidden_dropout_prob if config.project_dropout_prob is None else config.project_dropout_prob)
      self.project_act_fn = ACT2FN[config.project_act]

    self.config = config

  def forward(self, hidden_states, type_ids, input_mask):
    with torch.no_grad():
      attention_mask = input_mask.to(hidden_states.device)
      type_ids = type_ids.to(hidden_states.device)
      seq_len = attention_mask.sum(-1, keepdim=True)
      type_a_len = seq_len - (type_ids*attention_mask).sum(-1, keepdim=True)
      context_pos = torch.zeros_like(seq_len)
      # mask out special token
      #sep_idx = torch.cat((context_pos, type_a_len - 1, seq_len - 1), dim = -1)
      sep_idx = torch.cat((type_a_len - 1, seq_len - 1), dim = -1)
      attention_mask.scatter_(-1, sep_idx, 0)
      type_a = (1-type_ids)*attention_mask
      type_b = type_ids*attention_mask
      attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.device)

    pooled_output = self.attention(hidden_states[:, 0].unsqueeze(1), hidden_states, hidden_states, attention_mask, type_a, type_b).squeeze(1)
    if self.config.intermediate_size > 0:
      output = self.expand(pooled_output)
      output = self.intermediate_act_fn(output)
      if self.config.shrink_size > 0:
        output = self.shrink(output)
      if pooled_output.size(-1)==output.size(-1):
        output = MaskedLayerNorm(self.LayerNorm, self.dropout(output) + pooled_output)
    else:
      output = pooled_output
    if self.config.project_size > 0:
      output = self.project_act_fn(self.project(self.project_dropout(output)))
    return output

  def output_dim(self):
    if self.config.project_size > 0:
      return self.config.project_size
    elif self.config.shrink_size > 0:
      return self.config.shrink_size
    elif self.config.intermediate_size > 0:
      return self.config.intermediate_size
    else:
      return self.config.hidden_size

