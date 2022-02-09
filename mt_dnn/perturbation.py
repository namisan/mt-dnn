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
from .loss import stable_kl

logger = logging.getLogger(__name__)


def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


class SmartPerturbation:
    def __init__(
        self,
        epsilon=1e-6,
        multi_gpu_on=False,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p="inf",
        k=1,
        fp16=False,
        encoder_type=EncoderModelType.BERT,
        loss_map=[],
        norm_level=0,
    ):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.encoder_type = encoder_type
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0

    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == "l2":
            if sentence_level:
                direction = grad / (
                    torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon
                )
            else:
                direction = grad / (
                    torch.norm(grad, dim=-1, keepdim=True) + self.epsilon
                )
        elif self.norm_p == "l1":
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (
                    grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon
                )
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (
                    grad.abs().max(-1, keepdim=True)[0] + self.epsilon
                )
        return direction, eff_direction

    def forward(
        self,
        model,
        logits,
        input_ids,
        token_type_ids,
        attention_mask,
        premise_mask=None,
        hyp_mask=None,
        task_id=0,
        task_type=TaskType.Classification,
        pairwise=1,
    ):
        # adv training
        assert task_type in set(
            [TaskType.Classification, TaskType.Ranking, TaskType.Regression]
        ), "Donot support {} yet".format(task_type)
        vat_args = [
            input_ids,
            token_type_ids,
            attention_mask,
            premise_mask,
            hyp_mask,
            task_id,
            1,
        ]

        # init delta
        embed = model(*vat_args)
        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)
        for step in range(0, self.K):
            vat_args = [
                input_ids,
                token_type_ids,
                attention_mask,
                premise_mask,
                hyp_mask,
                task_id,
                2,
                embed + noise,
            ]
            adv_logits = model(*vat_args)
            if task_type == TaskType.Regression:
                adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction="sum")
            else:
                if task_type == TaskType.Ranking:
                    adv_logits = adv_logits.view(-1, pairwise)
                adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            (delta_grad,) = torch.autograd.grad(
                adv_loss, noise, only_inputs=True, retain_graph=False
            )
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            noise, eff_noise = self._norm_grad(
                delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level
            )
            noise = noise.detach()
            noise.requires_grad_()
        vat_args = [
            input_ids,
            token_type_ids,
            attention_mask,
            premise_mask,
            hyp_mask,
            task_id,
            2,
            embed + noise,
        ]
        adv_logits = model(*vat_args)
        if task_type == TaskType.Ranking:
            adv_logits = adv_logits.view(-1, pairwise)
        adv_lc = self.loss_map[task_id]
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        return adv_loss, embed.detach().abs().mean(), eff_noise.detach().abs().mean()
