# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import copy
import imp
import sys
import torch
import tasks
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from mt_dnn.loss import LOSS_REGISTRY
from mt_dnn.matcher import SANBertNetwork
from mt_dnn.perturbation import SmartPerturbation
from mt_dnn.loss import *
from mt_dnn.optim import AdamaxW
from data_utils.task_def import TaskType, EncoderModelType
from experiments.exp_def import TaskDef
from data_utils.my_statics import DUMPY_STRING_FOR_EMPTY_ANS


logger = logging.getLogger(__name__)


class MTDNNModel(object):
    def __init__(self, opt, device=None, state_dict=None, num_train_step=-1, tokenizer=None):
        self.tokenizer =tokenizer
        self.config = opt
        self.updates = (
            state_dict["updates"] if state_dict and "updates" in state_dict else 0
        )
        self.local_updates = 0
        self.device = device
        self.train_loss = AverageMeter()
        self.adv_loss = AverageMeter()
        self.emb_val = AverageMeter()
        self.eff_perturb = AverageMeter()
        self.initial_from_local = True if state_dict else False
        # if self.config["local_rank"] not in [-1, 0]:
        model = SANBertNetwork(opt, initial_from_local=self.initial_from_local)
        self.total_param = sum(
            [p.nelement() for p in model.parameters() if p.requires_grad]
        )
        if opt["cuda"] and (not self.config["deepspeed"]):
            model = model.to(self.device)
        self.network = model
        if state_dict:
            if 'state' in state_dict:
                missing_keys, unexpected_keys = self.network.load_state_dict(
                    state_dict["state"], strict=False
                )
                print("missing_keys:{}".format(missing_keys))
                print(unexpected_keys)
            else:
                missing_keys, unexpected_keys = self.network.bert.load_state_dict(
                    state_dict, strict=False
                )
                # print("missing_keys:{}".format(missing_keys))
                # print(unexpected_keys)
        optimizer_parameters = self._get_param_groups()

        self._setup_optim(optimizer_parameters, state_dict, num_train_step)
        self.optimizer.zero_grad()

        # if self.config["local_rank"] not in [-1, 0]:
        #    torch.distributed.barrier()
        if self.config["deepspeed"]:
            pass
        elif self.config["local_rank"] != -1:
            self.mnetwork = torch.nn.parallel.DistributedDataParallel(
                self.network,
                device_ids=[self.config["local_rank"]],
                output_device=self.config["local_rank"],
                find_unused_parameters=self.config.get("not_find_unused_parameters", False), # turn on during debugging
            )
        elif self.config["multi_gpu_on"]:
            self.mnetwork = nn.DataParallel(self.network)
        else:
            self.mnetwork = self.network
        self._setup_lossmap(self.config)
        self._setup_kd_lossmap(self.config)
        self._setup_adv_lossmap(self.config)
        self._setup_adv_training(self.config)

    def _setup_adv_training(self, config):
        self.adv_teacher = None
        if config.get("adv_train", False):
            self.adv_teacher = SmartPerturbation(
                config["adv_epsilon"],
                config["multi_gpu_on"],
                config["adv_step_size"],
                config["adv_noise_var"],
                config["adv_p_norm"],
                config["adv_k"],
                config["fp16"],
                config["encoder_type"],
                loss_map=self.adv_task_loss_criterion,
                norm_level=config["adv_norm_level"],
            )

    def _get_param_groups(self):
        no_decay = ["bias", "gamma", "beta", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    def _setup_optim(self, optimizer_parameters, state_dict=None, num_train_step=-1):
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optimizer_parameters, self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = AdamaxW(optimizer_parameters,
                                    lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(optimizer_parameters,
                                    lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])


        if state_dict and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.config["fp16"] and not self.config["deepspeed"]:
            try:
                from apex import amp
                global amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(
                self.network, self.optimizer, opt_level=self.config["fp16_opt_level"]
            )
            self.network = model
            self.optimizer = optimizer

        # # set up scheduler
        self.scheduler = None
        scheduler_type = self.config['scheduler_type']
        warmup_steps = self.config['warmup'] * num_train_step
        if scheduler_type == 3:
            from transformers import get_polynomial_decay_schedule_with_warmup
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )
        if scheduler_type == 2:
            from transformers import get_constant_schedule_with_warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps
                )
        elif scheduler_type == 1:
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )

        if self.config["deepspeed"]:
            import deepspeed
            import json
            with open(self.config["deepspeed_config"], 'r') as fd:
                ds_config = json.load(fd)
            network, optimizer, _, _ = deepspeed.initialize(
                                        model=self.network,
                                        optimizer=self.optimizer,
                                        model_parameters=optimizer_parameters,
                                        config=ds_config,
            )
            self.optimizer = optimizer
            self.mnetwork = network

    def _setup_lossmap(self, config):
        task_def_list: List[TaskDef] = config["task_def_list"]
        self.task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.loss
            lc = LOSS_REGISTRY[cs](name="Loss func of task {}: {}".format(idx, cs))
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self, config):
        task_def_list: List[TaskDef] = config["task_def_list"]
        self.kd_task_loss_criterion = []
        if config.get("mkd_opt", 0) > 0:
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.kd_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](
                    name="KD Loss func of task {}: {}".format(idx, cs)
                )
                self.kd_task_loss_criterion.append(lc)

    def _setup_adv_lossmap(self, config):
        task_def_list: List[TaskDef] = config["task_def_list"]
        self.adv_task_loss_criterion = []
        if config.get("adv_train", False):
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.adv_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](
                    name="Adv Loss func of task {}: {}".format(idx, cs)
                )
                self.adv_task_loss_criterion.append(lc)

    def _to_cuda(self, tensor):
        if tensor is None:
            return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            # y = [e.cuda(non_blocking=True) for e in tensor]
            y = [e.to(self.device) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            # y = tensor.cuda(non_blocking=True)
            y = tensor.to(self.device)
            y.requires_grad = False
        return y

    def update(self, batch_meta, batch_data):
        self.network.train()
        y = batch_data[batch_meta["label"]]
        y = self._to_cuda(y) if self.config["cuda"] else y
        if batch_meta["task_def"]["task_type"] == TaskType.SeqenceGeneration:
            y = y.view(-1)

        task_id = batch_meta["task_id"]
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        if "y_token_id" in batch_meta:
            inputs.append(batch_data[batch_meta["y_token_id"]])
        weight = None
        if self.config.get("weighted_on", False):
            if self.config["cuda"]:
                weight = batch_data[batch_meta["factor"]].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta["factor"]]

        # fw to get logits
        logits = self.mnetwork(*inputs)

        ignore_index = batch_meta.get("ignore_index", -1)

        # compute loss
        loss = 0
        if self.task_loss_criterion[task_id] and (y is not None):
            loss_criterion = self.task_loss_criterion[task_id]
            if (
                isinstance(loss_criterion, RankCeCriterion)
                and batch_meta["pairwise_size"] > 1
            ):
                # reshape the logits for ranking.
                loss = self.task_loss_criterion[task_id](
                    logits,
                    y,
                    weight,
                    ignore_index=ignore_index,
                    pairwise_size=batch_meta["pairwise_size"],
                )
            else:
                loss = self.task_loss_criterion[task_id](
                    logits, y, weight, ignore_index=ignore_index
                )

        # compute kd loss
        if self.config.get("mkd_opt", 0) > 0 and ("soft_label" in batch_meta):
            soft_labels = batch_meta["soft_label"]
            soft_labels = (
                self._to_cuda(soft_labels) if self.config["cuda"] else soft_labels
            )
            kd_lc = self.kd_task_loss_criterion[task_id]
            kd_loss = (
                kd_lc(logits, soft_labels, weight, ignore_index=ignore_index) if kd_lc else 0
            )
            loss = loss + kd_loss

        # adv training
        if self.config.get("adv_train", False) and self.adv_teacher:
            # task info
            task_type = batch_meta["task_def"]["task_type"]
            adv_inputs = (
                [self.mnetwork, logits]
                + inputs
                + [task_type, batch_meta.get("pairwise_size", 1)]
            )
            adv_loss, emb_val, eff_perturb = self.adv_teacher.forward(*adv_inputs)
            loss = loss + self.config["adv_alpha"] * adv_loss

        batch_size = batch_data[batch_meta["token_id"]].size(0)
        # rescale loss as dynamic batching
        if self.config["bin_on"]:
            loss = loss * (1.0 * batch_size / self.config["batch_size"])
        if self.config["local_rank"] != -1:
            # print('Rank ', self.config['local_rank'], ' loss ', loss)
            copied_loss = copy.deepcopy(loss.data)
            torch.distributed.all_reduce(copied_loss)
            copied_loss = copied_loss / self.config["world_size"]
            self.train_loss.update(copied_loss.item(), batch_size)
        else:
            self.train_loss.update(loss.item(), batch_size)

        if self.config.get("adv_train", False) and self.adv_teacher:
            if self.config["local_rank"] != -1:
                copied_adv_loss = copy.deepcopy(adv_loss.data)
                torch.distributed.all_reduce(copied_adv_loss)
                copied_adv_loss = copied_adv_loss / self.config["world_size"]
                self.adv_loss.update(copied_adv_loss.item(), batch_size)

                copied_emb_val = copy.deepcopy(emb_val.data)
                torch.distributed.all_reduce(copied_emb_val)
                copied_emb_val = copied_emb_val / self.config["world_size"]
                self.emb_val.update(copied_emb_val.item(), batch_size)

                copied_eff_perturb = copy.deepcopy(eff_perturb.data)
                torch.distributed.all_reduce(copied_eff_perturb)
                copied_eff_perturb = copied_eff_perturb / self.config["world_size"]
                self.eff_perturb.update(copied_eff_perturb.item(), batch_size)
            else:
                self.adv_loss.update(adv_loss.item(), batch_size)
                self.emb_val.update(emb_val.item(), batch_size)
                self.eff_perturb.update(eff_perturb.item(), batch_size)

        # scale loss
        loss = loss / self.config.get("grad_accumulation_step", 1)
        if self.config["deepspeed"]:
            self.mnetwork.backward(loss)
        elif self.config["fp16"]:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.get("grad_accumulation_step", 1) == 0:
            if self.config["global_grad_clipping"] > 0:
                if self.config["fp16"] and not self.config["deepspeed"]:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer),
                        self.config["global_grad_clipping"],
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config["global_grad_clipping"]
                    )
            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            if self.scheduler:
                self.scheduler.step()

    def encode(self, batch_meta, batch_data):
        self.network.eval()
        inputs = batch_data[:3]
        sequence_output = self.network.encode(*inputs)[0]
        return sequence_output

    # TODO: similar as function extract, preserve since it is used by extractor.py
    # will remove after migrating to transformers package
    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def predict(self, batch_meta, batch_data):
        self.network.eval()
        task_id = batch_meta["task_id"]
        task_def = TaskDef.from_dict(batch_meta["task_def"])
        task_type = task_def.task_type
        task_obj = tasks.get_task_obj(task_def)
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        if task_type == TaskType.SeqenceGeneration:
            # y_idx, #3 -> gen
            inputs.append(None)
            inputs.append(3)

        score = self.mnetwork(*inputs)

        if task_type == TaskType.SeqenceLabeling:
            batch_meta["mask"] = batch_data[batch_meta["mask"]]

        if task_obj is not None:
            return task_obj.test_predict(score, batch_meta, tokenizer=self.tokenizer)
            
        elif task_type == TaskType.SeqenceGenerationMRC:
            predicts = self.tokenizer.batch_decode(score, skip_special_tokens=True)
            predictions = {}
            golds = {}
            for idx, predict in enumerate(predicts):
                sample_id = batch_meta["uids"][idx]
                answer = batch_meta["answer"][idx]
                predict = predict.strip()
                if predict == DUMPY_STRING_FOR_EMPTY_ANS:
                    predict = ""
                predictions[sample_id] = predict
                golds[sample_id] = answer
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy().tolist()
            return score, predictions, golds
        elif task_type == TaskType.ClozeChoice:
            score = score.contiguous().view(-1)
            score = score.data.cpu()
            score = score.numpy()
            copy_score = score.tolist()
            answers = batch_meta["answer"]
            choices = batch_meta["choice"]
            chunks = batch_meta["pairwise_size"]
            uids = batch_meta["uids"]
            predictions = {}
            golds = {}
            for chunk in chunks:
                uid = uids[0]
                answer = eval(answers[0])
                choice = eval(choices[0])
                answers = answers[chunk:]
                choices = choices[chunk:]
                current_p = score[:chunk]
                score = score[chunk:]
                positive = np.argmax(current_p)
                predict = choice[positive]
                predictions[uid] = predict
                golds[uid] = answer
            return copy_score, predictions, golds            
        else:
            raise ValueError("Unknown task_type: %s" % task_type)
        return score, predict, batch_meta["label"]

    def save(self, filename):
        if isinstance(self.mnetwork, torch.nn.parallel.DistributedDataParallel):
            model = self.mnetwork.module
        else:
            model = self.network
        # network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        network_state = dict([(k, v.cpu()) for k, v in model.state_dict().items()])
        params = {
            "state": network_state,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(params, filename)
        logger.info("model saved to {}".format(filename))

    def load(self, checkpoint):
        model_state_dict = torch.load(checkpoint)
        if "state" in model_state_dict:
            self.network.load_state_dict(model_state_dict["state"], strict=False)
        if "optimizer" in model_state_dict:
            self.optimizer.load_state_dict(model_state_dict["optimizer"])
        if "config" in model_state_dict:
            self.config.update(model_state_dict["config"])

    def cuda(self):
        self.network.cuda()
