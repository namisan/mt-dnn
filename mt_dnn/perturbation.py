# Copyright (c) Microsoft. All rights reserved.
from copy import deepcopy
import torch
import logging
import random
from torch.nn import Parameter
from functools import wraps
import torch.nn.functional as F
from data_utils.task_def import TaskType
from data_utils.task_def import EncoderModelType

logger = logging.getLogger(__name__)

def generate_noise(embed, mask, epsilon=1e-5, encoder_type=EncoderModelType.ROBERTA):
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    if encoder_type == EncoderModelType.ROBERTA:
        embedding_mask = 1 - mask.unsqueeze(-1).type_as(embed)
        newembed = embed * embedding_mask
        noise = noise * embedding_mask
    else:
        newembed = (embed.data.detach()+ noise).detach()
        embedding_mask = mask.unsqueeze(2).type_as(embed)
        newembed = embed * embedding_mask
        noise = noise * embedding_mask
    noise.detach()
    noise.requires_grad_()
    return newembed, noise


class SmartPerturbation():
    def __init__(self,
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='inf',
                 fp16=False,
                 encoder_type=EncoderModelType.BERT,
                 loss_map=[]):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon 
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        # sigma
        self.noise_var = noise_var 
        self.norm_p = norm_p
        self.encoder_type = encoder_type 
        self.loss_map = loss_map 
        assert len(loss_map) > 0


    def _norm_grad(self, grad):
        if self.norm_p == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction

    def forward(self, model,
                logits,
                input_ids,
                token_type_ids,
                attention_mask,
                premise_mask=None,
                hyp_mask=None,
                task_id=0,
                task_type=TaskType.Classification,
                pairwise=1):
        # adv training
        assert task_type in set([TaskType.Classification, TaskType.Ranking, TaskType.Regression]), 'Donot support {} yet'.format(task_type)
        vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 1]
        embed = model(*vat_args)
        embed, delta = generate_noise(embed, attention_mask, epsilon=self.noise_var, encoder_type=self.encoder_type)
        vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, embed + delta]
        adv_logits = model(*vat_args)
        if task_type == TaskType.Regression:
            adv_loss = F.mse_loss(adv_logits, logits)
        else:
            if task_type == TaskType.Ranking:
                adv_logits = adv_logits.view(-1, pairwise)
            adv_loss = F.kl_div(F.log_softmax(adv_logits, dim=-1, dtype=torch.float32), F.softmax(logits.detach(), dim=-1, dtype=torch.float32), reduction='batchmean')
        delta_grad, = torch.autograd.grad(adv_loss, delta, only_inputs=True)
        norm = delta_grad.norm()
        if (torch.isnan(norm) or torch.isinf(norm)):
            return 0
        delta_grad = self._norm_grad(delta_grad)
        embed = embed + delta_grad * self.step_size
        embed = embed.detach()
        vat_args = [input_ids, token_type_ids, attention_mask, premise_mask, hyp_mask, task_id, 2, embed]
        adv_logits = model(*vat_args)
        if task_type == TaskType.Ranking:
            adv_logits = adv_logits.view(-1, pairwise)
        adv_lc = self.loss_map[task_id]
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        return adv_loss 
