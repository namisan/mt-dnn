#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#
""" Masked language model
"""
import torch
from torch import nn

def random_mask_input(input_ids, vocab_dict, mask_ratio=0.15, keep_ratio=0.1, other_ratio=0.1):
  if mask_ratio<0.001:
    return input_ids
  special_tok=['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']
  start_id = 1000
  end_id = len(vocab_dict)
  replace_ids = torch.randint_like(input_ids, start_id, end_id)
  keep_ids = torch.zeros_like(input_ids).bernoulli_(keep_ratio/(keep_ratio + other_ratio))
  replace_ids = torch.where(keep_ids>0, input_ids, replace_ids)
  keep_ids = torch.zeros_like(input_ids).bernoulli_(1 - (keep_ratio + other_ratio))
  replace_ids = torch.where(keep_ids>0, torch.tensor([vocab_dict['[MASK]']]).to(replace_ids), replace_ids)
  mask_ids = torch.zeros_like(input_ids).bernoulli_(mask_ratio)
  mask_ids = torch.where(input_ids>=1000, mask_ids, torch.tensor(0).to(mask_ids))
  masked_inputs = torch.where(mask_ids>0, replace_ids, input_ids)
  masked_labels = torch.where(mask_ids>0, input_ids, torch.tensor(-1).to(input_ids))
  masked_inputs = torch.stack((masked_inputs, masked_labels), dim=1)
  return masked_inputs

class LMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(LMPredictionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        #self.transform_act_fn = ACT2FN[config.hidden_act] \
        #    if isinstance(config.hidden_act, str) else config.hidden_act
        #self.LayerNorm = BertLayerNorm(config.hidden_size, 1e-7)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
        #                         bert_model_embedding_weights.size(0),
        #                         bias=False)
        #self.decoder.weight = bert_model_embedding_weights.transpose(0,1)
        # v x d
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states, embeding_weight):
        # b x s x d
        #hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense(hidden_states)
        #hidden_states = self.transform_act_fn(hidden_states)

        # b x s x v 
        logits = torch.matmul(hidden_states, embeding_weight.t()) + self.bias
        return logits
