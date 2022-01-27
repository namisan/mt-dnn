# Copyright (c) Microsoft, Inc. 2020 
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion

def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss

def SKL(logit, target, epsilon=1e-8):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    #bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    return (p* (rp- ry) * 2).sum()


@register_criterion('adv_masked_lm')
class AdvMaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()
        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits, extra = model(**sample['net_input'], masked_tokens=masked_tokens, player=-1)
        targets = model.get_targets(sample, [logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        if self.args.adv_opt > 0 and self.training:
            embed = extra['inner_states'][self.args.prob_n_layer]
            noise = embed.data.new(embed.size()).normal_(0, 1) * self.args.noise_var
            noise.requires_grad_()
            newembed = embed.data.detach() + noise
            adv_logits, _ = model(**sample['net_input'], masked_tokens=masked_tokens, task_id=1, embed=newembed, player=0)
            adv_loss = KL(adv_logits, logits.detach(), reduction="batchmean")
            # line 5, g_adv
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                # skim this batch
                logging_output = {
                    'loss': utils.item(loss.data) if reduce else loss.data,
                    'nll_loss': utils.item(loss.data) if reduce else loss.data,
                    'ntokens': sample['ntokens'],
                    'nsentences': sample['nsentences'],
                    'sample_size': sample_size,
                }
                return loss, sample_size, logging_output
            # line 6 inner sum
            noise = noise + delta_grad * self.args.adv_step_size
            # line 6 projection
            noise = self.adv_project(noise, norm_type=self.args.project_norm_type, eps=self.args.noise_gamma)
            newembed = embed.data.detach() + noise
            newembed = newembed.detach()
            adv_logits, _ = model(**sample['net_input'], masked_tokens=masked_tokens, task_id=1, embed=newembed, player=0)
            # line 8 symmetric KL
            adv_loss_f = KL(adv_logits, logits.detach())
            adv_loss_b = KL(logits, adv_logits.detach())
            adv_loss = (adv_loss_f + adv_loss_b) * self.args.adv_alpha
            loss = loss + adv_loss
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output

