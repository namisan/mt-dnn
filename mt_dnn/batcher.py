# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import sys
import json
import torch
import random
import string
import logging
import numpy as np
import pickle as pkl
from shutil import copyfile

UNK_ID=100
BOS_ID=101

class BatchGen:
    def __init__(self, data, batch_size=32, gpu=True, is_train=True,
                 maxlen=128, dropout_w=0.005,
                 do_batch=True, weighted_on=False,
                 task_id=0,
                 pairwise=False,
                 task=None,
                 task_type=0,
                 data_type=0):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.is_train = is_train
        self.gpu = gpu
        self.weighted_on = weighted_on
        self.data = data
        self.task_id = task_id
        self.pairwise = pairwise
        self.pairwise_size = 1
        self.data_type = data_type
        self.task_type=task_type
        if do_batch:
            if is_train:
                indices = list(range(len(self.data)))
                random.shuffle(indices)
                data = [self.data[i] for i in indices]
            self.data = BatchGen.make_baches(data, batch_size)
        self.offset = 0
        self.dropout_w = dropout_w

    @staticmethod
    def make_baches(data, batch_size=32):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    @staticmethod
    def load(path, is_train=True, maxlen=128, factor=1.0, pairwise=False):
        with open(path, 'r', encoding='utf-8') as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                sample['factor'] = factor
                cnt += 1
                if is_train:
                    if pairwise and (len(sample['token_id'][0]) > maxlen or len(sample['token_id'][1]) > maxlen):
                        continue
                    if (not pairwise) and (len(sample['token_id']) > maxlen):
                        continue
                data.append(sample)
            print('Loaded {} samples out of {}'.format(len(data), cnt))
            return data

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
        self.offset = 0

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    def __len__(self):
        return len(self.data)

    def patch(self, v):
        v = v.cuda(async=True)
        return v

    @staticmethod
    def todevice(v, device):
        v = v.to(device)
        return v

    def rebacth(self, batch):
        newbatch = []
        for sample in batch:
            size = len(sample['token_id'])
            self.pairwise_size = size
            assert size == len(sample['type_id'])
            for idx in range(0, size):
                token_id = sample['token_id'][idx]
                type_id = sample['type_id'][idx]
                uid = sample['ruid'][idx]
                olab = sample['olabel'][idx]
                newbatch.append({'uid': uid, 'token_id': token_id, 'type_id': type_id, 'label':sample['label'], 'true_label': olab})
        return newbatch


    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            if self.pairwise:
                batch = self.rebacth(batch)
            batch_size = len(batch)
            batch_dict = {}
            tok_len = max(len(x['token_id']) for x in batch)
            hypothesis_len = max(len(x['type_id']) - sum(x['type_id']) for x in batch)
            token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)
            if self.data_type < 1:
                premise_masks = torch.ByteTensor(batch_size, tok_len).fill_(1)
                hypothesis_masks = torch.ByteTensor(batch_size, hypothesis_len).fill_(1)

            for i, sample in enumerate(batch):
                select_len = min(len(sample['token_id']), tok_len)
                tok = sample['token_id']
                if self.is_train:
                    tok = self.__random_select__(tok)
                token_ids[i, :select_len] = torch.LongTensor(tok[:select_len])
                type_ids[i, :select_len] = torch.LongTensor(sample['type_id'][:select_len])
                masks[i, :select_len] = torch.LongTensor([1] * select_len)
                if self.data_type < 1:
                    hlen = len(sample['type_id']) - sum(sample['type_id'])
                    hypothesis_masks[i, :hlen] = torch.LongTensor([0] * hlen)
                    for j in range(hlen, select_len):
                        premise_masks[i, j] = 0
            if self.data_type < 1:
                batch_info = {
                    'token_id': 0,
                    'segment_id': 1,
                    'mask': 2,
                    'premise_mask': 3,
                    'hypothesis_mask': 4
                    }
                batch_data = [token_ids, type_ids, masks, premise_masks, hypothesis_masks]
                current_idx = 5
                valid_input_len = 5
            else:
                batch_info = {
                    'token_id': 0,
                    'segment_id': 1,
                    'mask': 2
                    }
                batch_data = [token_ids, type_ids, masks]
                current_idx = 3
                valid_input_len = 3

            if self.is_train:
                labels = [sample['label'] for sample in batch]
                if self.task_type > 0:
                    batch_data.append(torch.FloatTensor(labels))
                else:
                    batch_data.append(torch.LongTensor(labels))
                batch_info['label'] = current_idx
                current_idx += 1

            if self.gpu:
                for i, item in enumerate(batch_data):
                    batch_data[i] = self.patch(item.pin_memory())

            # meta 
            batch_info['uids'] = [sample['uid'] for sample in batch]
            batch_info['task_id'] = self.task_id
            batch_info['input_len'] = valid_input_len
            batch_info['pairwise'] = self.pairwise
            batch_info['pairwise_size'] = self.pairwise_size
            batch_info['task_type'] = self.task_type
            if not self.is_train:
                labels = [sample['label'] for sample in batch]
                batch_info['label'] = labels
                if self.pairwise:
                    batch_info['true_label'] = [sample['true_label'] for sample in batch]
            self.offset += 1
            yield batch_info, batch_data
