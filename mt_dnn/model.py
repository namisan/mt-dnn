# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from pytorch_pretrained_bert import BertAdam as Adam
from module.bert_optim import Adamax
from module.my_optim import EMA
from .matcher import SANBertNetwork

logger = logging.getLogger(__name__)

class MTDNNModel(object):
    def __init__(self, opt, state_dict=None, num_train_step=-1):
        self.config = opt
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.train_loss = AverageMeter()
        self.network = SANBertNetwork(opt)

        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['state'].keys()):
                if k not in new_state:
                    del state_dict['state'][k]
            for k, v in list(self.network.state_dict().items()):
                if k not in state_dict['state']:
                    state_dict['state'][k] = v
            self.network.load_state_dict(state_dict['state'])
        self.mnetwork = nn.DataParallel(self.network) if opt['multi_gpu_on'] else self.network
        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])

        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in self.network.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
            ]
        # note that adamax are modified based on the BERT code
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.sgd(optimizer_parameters, opt['learning_rate'],
                                       weight_decay=opt['weight_decay'])

        elif opt['optimizer'] == 'adamax':
            self.optimizer = Adamax(optimizer_parameters,
                                        opt['learning_rate'],
                                        warmup=opt['warmup'],
                                        t_total=num_train_step,
                                        max_grad_norm=opt['grad_clipping'],
                                        schedule=opt['warmup_schedule'])
            if opt.get('have_lr_scheduler', False): opt['have_lr_scheduler'] = False
        elif opt['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(optimizer_parameters,
                                            opt['learning_rate'],
                                            rho=0.95)
        elif opt['optimizer'] == 'adam':
            self.optimizer = Adam(optimizer_parameters,
                                        lr=opt['learning_rate'],
                                        warmup=opt['warmup'],
                                        t_total=num_train_step,
                                        max_grad_norm=opt['grad_clipping'],
                                        schedule=opt['warmup_schedule'])
            if opt.get('have_lr_scheduler', False): opt['have_lr_scheduler'] = False
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if opt.get('have_lr_scheduler', False):
            if opt.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=opt['lr_gamma'], patience=3)
            elif opt.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentialLR(self.optimizer, gamma=opt.get('lr_gamma', 0.95))
            else:
                milestones = [int(step) for step in opt.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=opt.get('lr_gamma'))
        else:
            self.scheduler = None
        self.ema = None
        if opt['ema_opt'] > 0:
            self.ema = EMA(self.config['ema_gamma'], self.network)
        self.para_swapped=False

    def setup_ema(self):
        if self.config['ema_opt']:
            self.ema.setup()

    def update_ema(self):
        if self.config['ema_opt']:
            self.ema.update()

    def eval(self):
        if self.config['ema_opt']:
            self.ema.swap_parameters()
            self.para_swapped = True

    def train(self):
        if self.para_swapped:
            self.ema.swap_parameters()
            self.para_swapped = False

    def update(self, batch_meta, batch_data):
        self.network.train()
        labels = batch_data[batch_meta['label']]
        if batch_meta['pairwise']:
            labels = labels.contiguous().view(-1, batch_meta['pairwise_size'])[:, 0]
        if self.config['cuda']:
            y = Variable(labels.cuda(async=True), requires_grad=False)
        else:
            y = Variable(labels, requires_grad=False)
        task_id = batch_meta['task_id']
        task_type = batch_meta['task_type']
        inputs = batch_data[:batch_meta['input_len']]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        logits = self.mnetwork(*inputs)
        if batch_meta['pairwise']:
            logits = logits.view(-1, batch_meta['pairwise_size'])

        if self.config.get('weighted_on', False):
            if self.config['cuda']:
                weight = Variable(batch_data[batch_meta['factor']].cuda(async=True))
            else:
                weight = Variable(batch_data[batch_meta['factor']])
            if task_type > 0:
                loss = torch.mean(F.mse_loss(logits.squeeze(), y, reduce=False) * weight)
            else:
                loss = torch.mean(F.cross_entropy(logits, y, reduce=False) * weight)
        else:
            if task_type > 0:
                loss = F.mse_loss(logits.squeeze(), y)
            else:
                loss = F.cross_entropy(logits, y)

        self.train_loss.update(loss.item(), logits.size(0))
        self.optimizer.zero_grad()

        loss.backward()
        if self.config['global_grad_clipping'] > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                          self.config['global_grad_clipping'])
        self.optimizer.step()
        self.updates += 1
        self.update_ema()

    def predict(self, batch_meta, batch_data):
        self.network.eval()
        task_id = batch_meta['task_id']
        task_type = batch_meta['task_type']
        inputs = batch_data[:batch_meta['input_len']]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        score = self.mnetwork(*inputs)
        if batch_meta['pairwise']:
            score = score.contiguous().view(-1, batch_meta['pairwise_size'])
            if task_type < 1:
                score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.zeros(score.shape, dtype=int)
            positive = np.argmax(score, axis=1)
            for idx, pos in enumerate(positive):
                predict[idx, pos] = 1
            predict = predict.reshape(-1).tolist()
            score = score.reshape(-1).tolist()
            return score, predict, batch_meta['true_label']
        else:
            if task_type < 1:
                score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).tolist()
            score = score.reshape(-1).tolist()
        return score, predict, batch_meta['label']

    def save(self, filename):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        ema_state = dict(
            [(k, v.cpu()) for k, v in self.ema.model.state_dict().items()]) if self.ema is not None else dict()
        params = {
            'state': network_state,
            'optimizer': self.optimizer.state_dict(),
            'ema': ema_state,
            'config': self.config,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def cuda(self):
        self.network.cuda()
        if self.config['ema_opt']:
            self.ema.cuda()
