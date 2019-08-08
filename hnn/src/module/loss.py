#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#

""" Losses
"""

import torch

def LabelSmooth(num_labels, smooth_ratio):
  background = torch.zeros(1, num_labels)
  background.fill_(smooth_ratio / (num_labels-1))
  return background

def ClassificationLoss(logits, labels, lmr = 0):
  label_num = logits.size(-1)
  if lmr > 0:
    bg = LabelSmooth(label_num, lmr)
    bg_ = bg.repeat(labels.size(0), 1).to(labels.device)
    labels = labels.long()
    bg_.scatter_(1, labels.view(-1).unsqueeze(1), 1 - lmr)
    loss_fct = torch.nn.KLDivLoss(reduction='sum')
    log_softmax = torch.nn.LogSoftmax(1)
    loss = loss_fct(log_softmax(logits.view(-1, label_num).float()), bg_)/logits.size(0)
  else:
    if labels.dim()>1 and labels.size(-1)==label_num:
      log_softmax = torch.nn.LogSoftmax(1)
      logx = log_softmax(logits.view(-1, label_num)).float()
      loss = labels.view(-1, label_num).float()*logx
      loss = -loss.sum()/logits.size(0)
    else:
      loss_fct = torch.nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, label_num).float(), labels.view(-1).long())
  return loss

